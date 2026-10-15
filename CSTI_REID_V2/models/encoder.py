"""ViT-B/16 and experimental DeiT-S, ResNet-50(+IBN) cross-backbone style hooks."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def adain(x, mean, std):
    mu = x.mean(1, keepdim=True)
    sd = x.var(1, keepdim=True, unbiased=False).add(1e-6).sqrt()
    return (x-mu) / sd * std[:, None] + mean[:, None]


class ViTEncoder(nn.Module):
    def __init__(self, height=256, width=128, domain_tokens=6, pretrained=False):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        self.net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None,
                            image_size=224 if pretrained else height)
        self.embed_dim, self.height, self.width = 768, height, width
        self.domain_tokens = nn.Parameter(torch.randn(1, domain_tokens, 768)*0.02)
        self.offset = 1 + domain_tokens
        original = self.net.encoder.pos_embedding.detach()
        side = math.isqrt(original.shape[1]-1)
        if side**2 != original.shape[1]-1:
            raise ValueError('Incompatible pretrained positional embedding')
        sp = original[:, 1:].transpose(1, 2).reshape(1, 768, side, side)
        sp = F.interpolate(sp, size=(height//16,width//16),mode='bicubic',align_corners=False)
        sp = sp.flatten(2).transpose(1, 2)
        self.position = nn.Parameter(torch.cat([original[:, :1],
            original.new_zeros(1, domain_tokens, 768), sp], 1))
        self.net.encoder.pos_embedding.requires_grad_(False)

    def forward(self,x,style_code=None,return_style=False):
        if tuple(x.shape[-2:]) != (self.height,self.width):
            raise ValueError('Unexpected input crop geometry')
        b=x.shape[0]
        patch = self.net.conv_proj(x).flatten(2).transpose(1,2)
        cls = self.net.class_token.expand(b,-1,-1)
        z = self.net.encoder.dropout(torch.cat([cls,self.domain_tokens.expand(b,-1,-1),patch],1)+self.position)
        moments=[]
        for i,block in enumerate(self.net.encoder.layers):
            z=block(z)
            if i<3:
                spatial=z[:,self.offset:]
                moments += [spatial.mean(1),spatial.var(1,unbiased=False).add(1e-6).sqrt()]
                if style_code is not None:
                    mu=style_code[:,2*i*768:(2*i+1)*768]
                    sd=style_code[:,(2*i+1)*768:(2*i+2)*768]
                    z=torch.cat([z[:,:self.offset],adain(spatial,mu,sd)],1)
        f=F.normalize(self.net.encoder.ln(z)[:,0],dim=-1)
        return (f,torch.cat(moments,-1)) if return_style else f


class TinyEncoder(nn.Module):
    def __init__(self,height=32,width=16,domain_tokens=2):
        super().__init__()
        self.embed_dim=16
        self.conv=nn.Conv2d(3,16,3,padding=1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(16,16),nn.GELU(),nn.LayerNorm(16)) for _ in range(3)])
        self.domain_tokens=nn.Parameter(torch.zeros(1,domain_tokens,16))
        self.ln=nn.LayerNorm(16)
    def forward(self,x,style_code=None,return_style=False):
        z=self.conv(x).flatten(2).transpose(1,2)
        moments=[]
        for i,block in enumerate(self.blocks):
            z=block(z)+self.domain_tokens.mean(1).unsqueeze(1)
            moments += [z.mean(1),z.var(1,unbiased=False).add(1e-6).sqrt()]
            if style_code is not None:
                mu=style_code[:,i*32:i*32+16]
                sd=style_code[:,i*32+16:i*32+32]
                z=adain(z,mu,sd)
        f=F.normalize(self.ln(z.mean(1)),dim=-1)
        return (f,torch.cat(moments,-1)) if return_style else f


class IBN(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.a=n//2
        self.inorm=nn.InstanceNorm2d(self.a,affine=True)
        self.bnorm=nn.BatchNorm2d(n-self.a)
    def forward(self,x):
        return torch.cat([self.inorm(x[:,:self.a]),self.bnorm(x[:,self.a:])],1)


class ResNetEncoder(nn.Module):
    """CNN style adaptation with CNN-specific learned context, not literal ViT tokens."""
    def __init__(self,height=256,width=128,domain_tokens=6,pretrained=False,ibn=False):
        super().__init__()
        from torchvision.models import resnet50,ResNet50_Weights
        if pretrained and ibn:
            raise ValueError('IBN needs an IBN-compatible checkpoint')
        self.net=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.embed_dim=2048
        self.domain_tokens=nn.Parameter(torch.zeros(1,domain_tokens,2048))
        self.context=nn.ModuleList([nn.Linear(2048,d) for d in [256,512,1024]])
        self.output_ln=nn.LayerNorm(2048)
        if ibn:
            for layer in (self.net.layer1,self.net.layer2,self.net.layer3):
                for block in layer:
                    block.bn1=IBN(block.bn1.num_features)
    def forward(self,x,style_code=None,return_style=False):
        n=self.net
        h=n.maxpool(n.relu(n.bn1(n.conv1(x))))
        moments=[]; offset=0
        for i,(layer,d) in enumerate(zip((n.layer1,n.layer2,n.layer3),(256,512,1024))):
            h=layer(h)+self.context[i](self.domain_tokens.mean(1)).view(1,d,1,1)
            mu=h.mean((2,3)); sd=h.var((2,3),unbiased=False).add(1e-6).sqrt()
            moments.extend([mu,sd])
            if style_code is not None:
                tm=style_code[:,offset:offset+d]
                ts=style_code[:,offset+d:offset+2*d]
                h=(h-mu[:,:,None,None])/sd[:,:,None,None]*ts[:,:,None,None]+tm[:,:,None,None]
            offset += 2*d
        f=F.normalize(self.output_ln(n.avgpool(n.layer4(h)).flatten(1)),dim=-1)
        return (f,torch.cat(moments,-1)) if return_style else f


class DeiTEncoder(nn.Module):
    def __init__(self,height=256,width=128,domain_tokens=6,pretrained=False):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError('Install timm for DeiT-S') from exc
        self.net=timm.create_model('deit_small_patch16_224',pretrained=pretrained,num_classes=0,
                                   img_size=(height,width))
        self.embed_dim=self.net.num_features
        self.domain_tokens=nn.Parameter(torch.randn(1,domain_tokens,self.embed_dim)*.02)
        self.dompos=nn.Parameter(torch.zeros(1,domain_tokens,self.embed_dim))
        self.offset=1+domain_tokens
    def forward(self,x,style_code=None,return_style=False):
        b=x.shape[0]
        patches=self.net.patch_embed(x)
        if patches.ndim==4: patches=patches.flatten(2).transpose(1,2)
        z=torch.cat([self.net.cls_token.expand(b,-1,-1),self.domain_tokens.expand(b,-1,-1),patches],1)
        pos=torch.cat([self.net.pos_embed[:,:1],self.dompos,self.net.pos_embed[:,1:]],1)
        z=self.net.pos_drop(z+pos)
        stats=[]; d=self.embed_dim
        for i,block in enumerate(self.net.blocks):
            z=block(z)
            if i<3:
                s=z[:,self.offset:]
                stats.extend([s.mean(1),s.var(1,unbiased=False).add(1e-6).sqrt()])
                if style_code is not None:
                    z=torch.cat([z[:,:self.offset],adain(s,style_code[:,2*i*d:(2*i+1)*d],
                                  style_code[:,(2*i+1)*d:(2*i+2)*d])],1)
        f=F.normalize(self.net.norm(z)[:,0],dim=-1)
        return (f,torch.cat(stats,-1)) if return_style else f


def build_encoder(backbone_name='vit_b_16',pretrained=False,num_domain_tokens=6,height=256,width=128,freeze_backbone=False,**kwargs):
    args=(height,width,num_domain_tokens,pretrained)
    if backbone_name=='vit_b_16': model=ViTEncoder(*args)
    elif backbone_name=='tiny': model=TinyEncoder(height,width,num_domain_tokens)
    elif backbone_name=='deit_s': model=DeiTEncoder(*args)
    elif backbone_name=='resnet50': model=ResNetEncoder(*args)
    elif backbone_name=='resnet50_ibn': model=ResNetEncoder(*args,ibn=True)
    else: raise ValueError(f'Unknown encoder {backbone_name}')
    if freeze_backbone: freeze_tta(model)
    return model


def freeze_tta(model):
    for p in model.parameters(): p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module,nn.LayerNorm):
            for p in module.parameters(recurse=False): p.requires_grad_(True)
    model.domain_tokens.requires_grad_(True)
    return model


def adaptation_groups(model,lr_ln,lr_tokens):
    freeze_tta(model)
    norm=[p for n,p in model.named_parameters() if p.requires_grad and 'domain_tokens' not in n]
    return [{'params':norm,'lr':lr_ln},{'params':[model.domain_tokens],'lr':lr_tokens}]
