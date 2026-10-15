"""Source-identity-disjoint cross-camera mAP for checkpoint selection."""
from collections import defaultdict
import numpy as np
import torch
from data.datasets import loader
from utils.metrics import score


def build_source_dev_retrieval(samples):
    by_id=defaultdict(list)
    for s in samples:by_id[s.pid].append(s)
    queries=[];gallery=[]
    for pid,group in by_id.items():
        cameras=sorted({x.cam for x in group})
        if len(cameras)<2:continue
        chosen=next(s for s in group if s.cam==cameras[0])
        queries.append(chosen)
        gallery.extend([s for s in group if s != chosen])
    if not queries or not gallery:
        return None
    return queries,gallery

@torch.no_grad()
def source_dev_map(model,dev_sources,model_config,device):
    was_training=model.training
    model.eval()
    results={}
    for name,dev in dev_sources.items():
        split=build_source_dev_retrieval(dev)
        if split is None:continue
        query,gallery=split
        vecs=[]
        for samples in [query,gallery]:
            blocks=[]
            for images,_,_ in loader(samples,batch=32,height=model_config['height'],width=model_config['width']):
                blocks.append(model(images.to(device)).detach().cpu())
            vecs.append(torch.cat(blocks))
        q,g=vecs
        sim=(q@g.T).numpy()
        try:
            metric=score(sim,[s.pid for s in query],[s.cam for s in query],
                         [s.pid for s in gallery],[s.cam for s in gallery])
        except ValueError:continue
        results[name]=metric['mAP']
    if was_training:model.train()
    if not results:return None,{}
    return sum(results.values())/len(results),results
