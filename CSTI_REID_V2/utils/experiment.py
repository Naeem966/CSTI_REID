import json,random
from pathlib import Path
import numpy as np
import torch,yaml

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(path) as f:cfg=yaml.safe_load(f)
    if not isinstance(cfg,dict) or 'model' not in cfg:raise ValueError('Missing config model')
    return cfg

def read_state(path,device='cpu'):
    return torch.load(path,map_location=device,weights_only=True)

def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,default=str),encoding='utf8')
