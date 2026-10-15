"""Post-hoc open-set cross-camera mAP / CMC (never backpropagated)."""
import numpy as np

def score(sim,qpid,qcam,gpid,gcam):
    sim=np.asarray(sim);qpid=np.asarray(qpid);qcam=np.asarray(qcam)
    gpid=np.asarray(gpid);gcam=np.asarray(gcam)
    if sim.shape!=(len(qpid),len(gpid)):raise ValueError('Inconsistent similarity shape')
    ap=[];cmcs=[];inps=[];invalid=0
    for i in range(len(qpid)):
        include=(gpid>=0)&~((gpid==qpid[i])&(gcam==qcam[i]))
        ranks=np.flatnonzero(include)[np.argsort(-sim[i,include],kind='stable')]
        hits=gpid[ranks]==qpid[i]
        if not hits.any():invalid+=1;continue
        ap.append((np.cumsum(hits)/(np.arange(len(hits))+1)*hits).sum()/hits.sum())
        cmcs.append(np.cumsum(hits).clip(max=1))
        last_positive=np.flatnonzero(hits)[-1]+1
        inps.append(float(hits.sum()/last_positive))
    if not ap:raise ValueError('No admissible cross-camera positive')
    return {'mAP':float(np.mean(ap)),'rank1':float(np.mean([r[0] for r in cmcs])),
            'rank5':float(np.mean([r[min(4,len(r)-1)] for r in cmcs])),
            'rank10':float(np.mean([r[min(9,len(r)-1)] for r in cmcs])),
            'mINP':float(np.mean(inps)),
            'valid_queries':len(ap),'excluded_queries':invalid}

def camera_macro(folds):
    return {k:float(np.mean([v[k] for v in folds])) for k in ('mAP','rank1')}
