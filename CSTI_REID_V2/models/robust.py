"""Optional identity-safe KL-robust style analysis. NOT a headline model component."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import minimize_scalar

def identity_filter(source,branches,threshold=.25):
    s=F.normalize(source.detach(),dim=-1)
    b=F.normalize(branches,dim=-1)
    delta=1-b@s
    return delta<=threshold,delta

def kl_robust_risk(losses,occupancy,radius):
    """Finite support max_{KL(p||pi)<=radius} E_p[loss]."""
    if radius<0 or not len(losses):raise ValueError('KL radius/slot count invalid')
    if losses.shape!=occupancy.shape or (occupancy<=0).any():raise ValueError('Invalid occupancy')
    pi=occupancy/occupancy.sum()
    if radius==0:return (pi*losses).sum()
    if len(losses)==1:return losses[0]
    v=losses.detach().double().cpu().numpy();q=pi.detach().double().cpu().numpy()
    candidates=np.flatnonzero(v>=v.max()-1e-12)
    if radius>=-np.log(q[candidates].sum()):return losses[candidates].max()
    def objective(loglam):
        lam=np.exp(loglam);z=np.log(q)+v/lam;m=z.max()
        return lam*(radius+m+np.log(np.exp(z-m).sum()))
    optimum=minimize_scalar(objective,bounds=(-14,10),method='bounded')
    adversary=torch.softmax(pi.detach().log()+losses.detach()/np.exp(optimum.x),dim=0)
    return (adversary*losses).sum()

def robust_anchor_loss(source,branches,occupancy,kl_radius=.05,identity_threshold=.25):
    valid,discrepancy=identity_filter(source,branches,identity_threshold)
    if not valid.any():return None # clean-source fallback
    return kl_robust_risk(discrepancy[valid],occupancy[valid],kl_radius)

def accept_source_drift(source_before,candidate_after,epsilon=.1):
    delta=1-F.cosine_similarity(source_before.detach(),candidate_after,dim=-1)
    return bool(delta.mean()<=epsilon),float(delta.detach().mean())
