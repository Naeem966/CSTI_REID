"""Paper IWBRE/RNCC/MUC/DPPM, causal-order query only source-free adaptation."""
import math
import time
from copy import deepcopy
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .encoder import adaptation_groups
EPS=1e-6


def cosdist(a,b):
    return 1-F.normalize(a,dim=-1)@F.normalize(b,dim=-1).T


class StyleBank:
    def __init__(self,capacity=32,tau=0.15):
        self.capacity,self.tau=capacity,tau
        self.slots,self.counts,self.ages=[],[],[]
        self.time=0
    def observe(self,s):
        s=s.detach().flatten().clone(); self.time+=1
        if not self.slots:
            self.slots=[s];self.counts=[1];self.ages=[self.time];return
        d=cosdist(s[None],torch.stack(self.slots))[0]; j=int(d.argmin())
        if float(d.min())>self.tau and self.capacity>1:
            if len(self.slots)==self.capacity:
                evict=min(range(1,len(self.slots)),key=lambda i:self.ages[i])
                for arr in (self.slots,self.counts,self.ages):arr.pop(evict)
            self.slots.append(s);self.counts.append(1);self.ages.append(self.time)
        else:self.counts[j]+=1
    def vectors(self,ref):
        return torch.stack(self.slots).to(device=ref.device,dtype=ref.dtype)
    def occupancy(self,ref):
        counts=ref.new_tensor(self.counts)+EPS
        return counts/counts.sum()
    def sample(self,style,n,tau,importance=True):
        slots=self.vectors(style)
        q=torch.softmax(-((style[None]-slots)**2).sum(-1)/tau,dim=0)
        ix=torch.multinomial(q.detach(),n,replacement=True)
        if importance:
            p=self.occupancy(style)
            w=torch.softmax((p[ix]+EPS).log()-(q[ix]+EPS).log(),dim=0)
        else:w=style.new_full((n,),1/n)
        return slots[ix],w


class DPPM:
    def __init__(self,cfg):
        self.capacity=cfg['prototype_capacity'];self.rgate=cfg['tau_reliability']
        self.cgate=cfg['tau_consistency'];self.novelty=cfg['tau_novelty']
        self.eta=cfg['prototype_ema'];self.etau=cfg['utility_ema']
        self.protos,self.utilities,self.birth=[],[],[];self.clock=0
    def snapshot(self,ref):
        return torch.stack(self.protos).to(ref.device,ref.dtype).detach().clone() if self.protos else ref.new_empty(0,0)
    def update(self,feature,r,c):
        z=F.normalize(feature.detach().flatten(),dim=0)
        r=float(r.detach()) if torch.is_tensor(r) else float(r)
        c=float(c.detach()) if torch.is_tensor(c) else float(c)
        self.clock+=1
        self.utilities=[x*(1-self.etau) for x in self.utilities]
        if r<=self.rgate or c<=self.cgate:return 'rejected'
        if self.protos:
            d=cosdist(z[None],torch.stack(self.protos))[0];idx=int(d.argmin())
            self.utilities[idx]+=self.etau*r
            if float(d[idx])<=self.novelty:
                self.protos[idx]=F.normalize((1-self.eta*r)*self.protos[idx]+self.eta*r*z,dim=0)
                return 'updated'
        if len(self.protos)==self.capacity:
            i=min(range(len(self.protos)),key=lambda j:(self.utilities[j],self.birth[j]))
            for ar in (self.protos,self.utilities,self.birth):ar.pop(i)
        self.protos.append(z);self.utilities.append(self.etau*r);self.birth.append(self.clock)
        return 'admitted'



class UnfilteredMemory(DPPM):
    """Ablation: unfiltered FIFO memory without DPPM gates, EMA or clustering."""
    def update(self,feature,r,c):
        z=F.normalize(feature.detach().flatten(),dim=0)
        self.clock+=1
        if len(self.protos)==self.capacity:
            for a in (self.protos,self.utilities,self.birth):a.pop(0)
        self.protos.append(z)
        self.utilities.append(0.0)
        self.birth.append(self.clock)
        return 'unfiltered_admission'

def rncc(branches):
    k=len(branches)
    if k<2:return branches.new_zeros(())
    sim=F.normalize(branches,dim=-1)@F.normalize(branches,dim=-1).T
    return 1-2*sim.triu(1).sum()/(k*(k-1))


def muc(feature,branches,prototypes,alpha,beta,tau):
    if len(prototypes)<2:raise ValueError('MUC requires >=2 prototypes')
    f=F.normalize(feature,dim=0)
    similarities=F.normalize(prototypes,dim=-1)@f
    p=torch.softmax(similarities/tau,dim=0)
    uid=-(p*(p+EPS).log()).sum()/math.log(len(prototypes))
    usty=(F.normalize(branches,dim=-1)-f[None]).square().sum(-1).mean()
    return torch.exp(-alpha*uid-beta*usty),p


def normalize_reliability(raw,bootstrap=False):
    if bootstrap:return raw.clamp(min=0)
    if len(raw)==1:return raw.clamp(0,1) # avoids degeneracy at B=1
    return raw/raw.detach().max().clamp_min(EPS)


@dataclass
class QueryCache:
    embedding:torch.Tensor
    branches:torch.Tensor
    style:torch.Tensor
    consistency:torch.Tensor
    rncc:torch.Tensor
    raw:torch.Tensor
    entropy:torch.Tensor|None=None
    alignment:torch.Tensor|None=None


class CSTIAdapter:
    def __init__(self,encoder,config,device='cpu',variant='full'):
        self.cfg,self.device,self.variant=config,torch.device(device),variant
        self.model=encoder.to(self.device).eval()
        self.source=deepcopy(encoder).to(self.device).eval()
        self.source.requires_grad_(False)
        self.styles=StyleBank(config['style_capacity'],config['tau_style'])
        self.memory=UnfilteredMemory(config) if variant=='no_dppm' else DPPM(config)
        self.optimizer=torch.optim.SGD(adaptation_groups(self.model,config['lr_layernorm'],
                                        config['lr_tokens']),momentum=config['momentum'])
        self.gallery=None;self.trace=[];self.batch=0

    @torch.no_grad()
    def cache_gallery(self,loader):
        feats=[];ids=[];cams=[]
        for img,pid,cam in loader:
            feats.append(self.source(img.to(self.device)).detach())
            ids.extend(int(x) for x in pid);cams.extend(int(x) for x in cam)
        if not feats:raise ValueError('Empty gallery')
        self.gallery=torch.cat(feats)
        return ids,cams

    def _query(self,img,pre):
        _,s=self.model(img,return_style=True)
        s=s[0];self.styles.observe(s)
        k=self.cfg['samples_per_query']
        if k<2:raise ValueError('Need at least two intervention branches')
        bank,w=self.styles.sample(s,k,self.cfg['tau_proposal'],
            importance=self.variant not in ('simple_sa','no_iwbre'))
        branches=torch.stack([self.model(img,style_code=style[None])[0] for style in bank])
        feature=(branches*w[:,None]).sum(0)
        rloss=rncc(branches);c=1-rloss
        if self.variant=='no_rncc':
            c=torch.ones_like(c)
        if len(pre)<2 or self.variant=='simple_sa':
            return QueryCache(feature,branches,s,c,rloss,c.clamp_min(0))
        if self.variant=='no_muc':
            similarity=F.normalize(pre,dim=-1)@F.normalize(feature,dim=0)
            dist=torch.softmax(similarity/self.cfg['tau_prototype'],dim=0)
            raw=c.clamp_min(0)
        else:
            raw,dist=muc(feature,branches,pre,self.cfg['alpha'],self.cfg['beta'],self.cfg['tau_prototype'])
        logp=(dist+EPS).log()
        ent=-(dist*logp).sum()
        f=F.normalize(feature,dim=0)
        nearest=int((F.normalize(pre,dim=-1)@f).argmax())
        align=-logp[nearest]
        return QueryCache(feature,branches,s,c,rloss,raw,ent,align)

    def _style_loss(self,cache):
        if len(self.styles.slots)<2:return cache[0].embedding.new_zeros(())
        bank=self.styles.vectors(cache[0].style).detach()
        q=torch.stack([torch.softmax(-((v.style[None]-bank)**2).sum(-1)/self.cfg['tau_proposal'],0)
                       for v in cache]).mean(0)
        return (q*((q+EPS).log()+math.log(len(bank)))).sum()

    def step(self,images):
        if self.cfg.get('profile_runtime',False) and self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
        start=time.perf_counter()
        x=images.to(self.device)
        if x.ndim!=4 or len(x)==0:raise ValueError('Expected nonempty BxCxHxW batch')
        self.batch+=1
        # All MUC scores use one fixed PRE-BATCH prototype snapshot.
        pre=self.memory.snapshot(x)
        entries=[self._query(x[i:i+1],pre) for i in range(len(x))]
        boot=len(pre)<2 or self.variant in ('simple_sa','no_muc')
        reliability=normalize_reliability(torch.stack([e.raw for e in entries]),boot)
        embeddings=F.normalize(torch.stack([e.embedding for e in entries]),dim=-1)
        ranks=None
        if self.gallery is not None:
            ranks=(embeddings.detach()@F.normalize(self.gallery,dim=-1).T).argsort(-1,descending=True).cpu()
        loss=torch.stack([e.rncc for e in entries]).mean()
        if self.variant=='no_rncc':loss=loss*0
        if len(pre)>=2 and self.variant!='simple_sa':
            loss+=self.cfg['lambda_entropy']*(reliability*torch.stack([e.entropy for e in entries])).mean()
            loss+=self.cfg['lambda_alignment']*(reliability*torch.stack([e.alignment for e in entries])).mean()
        if self.variant!='simple_sa':loss+=self.cfg['lambda_style']*self._style_loss(entries)
        update=self.batch>self.cfg['warmup_batches'] and (len(pre)>=2 or self.variant=='simple_sa')
        if update and loss.requires_grad:
            self.optimizer.zero_grad(set_to_none=True);loss.backward();self.optimizer.step()
        else:update=False
        # Crucial: mutate live memory only AFTER the pre-update rankings and optional SGD.
        actions=[self.memory.update(e.embedding,reliability[i],e.consistency)
                 for i,e in enumerate(entries)] if self.variant!='simple_sa' else ['disabled']*len(entries)
        if self.cfg.get('profile_runtime',False) and self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
        event={'batch':self.batch,'pre_snapshot':len(pre),'post_memory':len(self.memory.protos),
               'style_slots':len(self.styles.slots),'optimizer_step':update,
               'loss':float(loss.detach()),'actions':actions,
               'batch_time_ms':1000*(time.perf_counter()-start),
               'gpu_memory_mib':(torch.cuda.memory_allocated(self.device)/1048576
                                 if self.device.type=='cuda' else None)}
        self.trace.append(event)
        return {'features':embeddings.detach().cpu(),'ranks':ranks,'reliability':reliability.detach().cpu(),'event':event}
