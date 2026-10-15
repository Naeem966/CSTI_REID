"""Source-only ID cross entropy + soft-margin batch-hard triplet."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class IdentityHead(nn.Module):
    def __init__(self,d,n):
        super().__init__();self.head=nn.Linear(d,n,bias=False)
    def forward(self,x):return self.head(x)

def soft_triplet(z,pid):
    dist=torch.cdist(F.normalize(z,dim=-1),F.normalize(z,dim=-1))
    equal=pid[:,None]==pid[None]
    same=equal.clone();same.fill_diagonal_(False)
    diff=~equal;valid=same.any(1)&diff.any(1)
    if not valid.any():return z.sum()*0
    hardp=dist.masked_fill(~same,float('-inf')).max(1).values[valid]
    hardn=dist.masked_fill(~diff,float('inf')).min(1).values[valid]
    return F.softplus(hardp-hardn).mean()

class SourceTrainer:
    def __init__(self,encoder,id_head,config,device='cpu',style_pool=None):
        self.encoder,self.head,self.cfg,self.device=encoder.to(device),id_head.to(device),config,device
        self.opt=torch.optim.AdamW(list(encoder.parameters())+list(id_head.parameters()),
                                   lr=config['lr'],weight_decay=config['weight_decay'])
        self.schedule=None
        self.style_pool=style_pool.to(device) if style_pool is not None else None
        self.style_p=float(config.get('style_intervention_probability',0.5))
    def setup_schedule(self,epochs):
        lo=self.cfg['lr_min']/self.cfg['lr']
        self.schedule=torch.optim.lr_scheduler.LambdaLR(self.opt,
            lambda e:lo+(1-lo)*(1+math.cos(math.pi*e/epochs))/2)
    def epoch(self,loader):
        self.encoder.train();self.head.train()
        totals=[0.,0.,0.];count=0
        for x,pid,_ in loader:
            x,pid=x.to(self.device),pid.to(self.device)
            # Source-only shallow-statistic substitution. Protocol A can supply
            # an ImageNet-derived style bank; Protocol B/C use current SOURCE
            # images only. The target is never loaded during source training.
            if self.style_p>0 and torch.rand(())<self.style_p:
                if self.style_pool is None:
                    with torch.no_grad():
                        _clean, source_styles=self.encoder(x,return_style=True)
                        perm=torch.randperm(len(source_styles),device=x.device)
                        sampled=source_styles[perm].detach()
                else:
                    idx=torch.randint(len(self.style_pool),(len(x),),device=x.device)
                    sampled=self.style_pool[idx].detach()
                z=self.encoder(x,style_code=sampled)
            else:
                z=self.encoder(x)
            ce=F.cross_entropy(self.head(z),pid,label_smoothing=self.cfg['label_smoothing'])
            tri=soft_triplet(z,pid);loss=ce+tri
            self.opt.zero_grad(set_to_none=True);loss.backward();self.opt.step()
            for i,part in enumerate((loss,ce,tri)):totals[i]+=float(part.detach())
            count+=1
        if not count:raise ValueError('No source training batches')
        if self.schedule:self.schedule.step()
        return {'loss':totals[0]/count,'ce':totals[1]/count,'triplet':totals[2]/count}
