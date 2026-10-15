"""Auditable identity-disjoint CSV ReID partitions; native Market/Duke parsing."""
import csv,random,re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset,DataLoader,Sampler
from torchvision import transforms as T

@dataclass(frozen=True)
class Sample:
    path:str
    pid:int
    cam:int
    dataset:str=''

@dataclass
class Splits:
    train:list
    query:list
    gallery:list
    name:str


def identity_audit(ds):
    overlap={x.pid for x in ds.train}&{x.pid for x in ds.query+ds.gallery}
    if overlap:raise ValueError(f'Train/test identity leakage: {sorted(overlap)[:5]}')

def load_reid(name,manifest=None,root=None):
    if manifest:
        path=Path(manifest);splits={'train':[],'query':[],'gallery':[]}
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                split=row['split'].lower().strip()
                if split not in splits:raise ValueError(f'Invalid split {split}')
                img=Path(row['path']);img=img if img.is_absolute() else path.parent/img
                splits[split].append(Sample(str(img),int(row['pid']),int(row['cam']),name))
        ds=Splits(**splits,name=name)
    elif root and name in ('market1501','dukemtmc'):
        parts=[]
        for folder in ('bounding_box_train','query','bounding_box_test'):
            d=Path(root)/folder
            if not d.is_dir():raise FileNotFoundError(d)
            samples=[]
            for p in sorted(d.glob('*.jpg')):
                m=re.match(r'^(-?\d+)_c(\d+)',p.name)
                if m and int(m[1])>=0:samples.append(Sample(str(p),int(m[1]),int(m[2]),name))
            parts.append(samples)
        ds=Splits(*parts,name=name)
    else:raise ValueError(f'{name}: supply official CSV manifest with path,pid,cam,split')
    if not all((ds.train,ds.query,ds.gallery)):raise ValueError('Empty required dataset split')
    identity_audit(ds)
    return ds

def heldout_ids(samples,fraction=.15,seed=42):
    ids=sorted({s.pid for s in samples})
    if len(ids)<2:raise ValueError('Need >=2 source identities')
    random.Random(seed).shuffle(ids)
    held=set(ids[:max(1,min(len(ids)-1,round(len(ids)*fraction)))])
    return [s for s in samples if s.pid not in held],[s for s in samples if s.pid in held]

def relabel_sources(groups):
    mapping={};out=[]
    for group in groups:
        for s in group:
            key=(s.dataset,s.pid)
            if key not in mapping:mapping[key]=len(mapping)
            out.append(Sample(s.path,mapping[key],s.cam,s.dataset))
    return out,len(mapping)

def b_outer(names=('market1501','dukemtmc','cuhk03','msmt17')):
    return {target:[n for n in names if n!=target] for target in names}

def b_inner(target):
    permitted=b_outer()[target]
    return {inner:[s for s in permitted if s!=inner] for inner in permitted}

def c_fold(ds,camera):
    identity_audit(ds)
    source=[x for x in ds.train if x.cam!=camera]
    query=[x for x in ds.query if x.cam==camera and any(g.pid==x.pid and g.cam!=x.cam for g in ds.gallery)]
    if not source or not query:raise ValueError(f'Camera {camera} has no eligible source/target')
    if {x.pid for x in source}&{q.pid for q in query}:raise ValueError('Protocol C identity leakage')
    return source,query,list(ds.gallery)

def transforms(train=False,height=256,width=128):
    ops=[T.Resize((height,width))]
    if train:ops += [T.RandomHorizontalFlip(),T.Pad(10),T.RandomCrop((height,width))]
    ops += [T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]
    if train:ops += [T.RandomErasing(p=.5)]
    return T.Compose(ops)

class ReIDDataset(Dataset):
    def __init__(self,samples,transform):self.samples=list(samples);self.transform=transform
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i]
        with Image.open(s.path) as im:x=self.transform(im.convert('RGB'))
        return x,s.pid,s.cam

class PKSampler(Sampler):
    def __init__(self,samples,P,K,seed=0):
        self.groups=defaultdict(list)
        for i,s in enumerate(samples):self.groups[s.pid].append(i)
        self.P,self.K,self.seed,self.epoch=P,K,seed,0
        if len(self.groups)<P:raise ValueError('Insufficient source IDs for PK batches')
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return len(self.groups)//self.P*self.P*self.K
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch)
        ids=list(self.groups);rng.shuffle(ids)
        for begin in range(0,len(ids)-self.P+1,self.P):
            for pid in ids[begin:begin+self.P]:
                x=self.groups[pid]
                yield from (rng.sample(x,self.K) if len(x)>=self.K else rng.choices(x,k=self.K))

def loader(samples,batch=32,train=False,height=256,width=128,P=None,K=None,workers=0,seed=0):
    sampler=PKSampler(samples,P,K,seed) if P is not None else None
    return DataLoader(ReIDDataset(samples,transforms(train,height,width)),batch_size=batch,
            sampler=sampler,shuffle=train and sampler is None,drop_last=train,
            num_workers=workers,pin_memory=False)
