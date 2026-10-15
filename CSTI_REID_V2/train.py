"""Leakage-checked source Re-ID training. Target data never enters training."""
import argparse
from pathlib import Path
import torch
from data.datasets import load_reid,heldout_ids,relabel_sources,loader,b_outer
from models.encoder import build_encoder
from models.source_trainer import SourceTrainer,IdentityHead
from utils.experiment import load_config,set_seed,write_json
from utils.source_validation import source_dev_map

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/paper.yaml')
    p.add_argument('--source-name',action='append',required=True)
    p.add_argument('--source-manifest',action='append',required=True)
    p.add_argument('--exclude-camera',type=int)
    p.add_argument('--outer-target')
    p.add_argument('--epochs',type=int)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--save-dir',default='checkpoints/source')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--source-weights',help='Preexisting architecture-compatible encoder state dict')
    p.add_argument('--source-style-pool',help='Optional .pt tensor/dict: permitted source-only style codes, [N,d_s]')
    args=p.parse_args()
    if len(args.source_name)!=len(args.source_manifest) or len(set(args.source_name))!=len(args.source_name):
        raise ValueError('Provide unique source name per manifest')
    if args.outer_target:
        if args.outer_target not in b_outer():raise ValueError('Unknown Protocol-B outer target')
        if set(args.source_name)!=set(b_outer()[args.outer_target]):
            raise ValueError('Protocol B requires exactly the three permitted sources')
    if args.exclude_camera is not None and len(args.source_name)!=1:
        raise ValueError('Protocol C requires camera-specific within-dataset source training')
    set_seed(args.seed)
    cfg=load_config(args.config);m=cfg['model'];c=cfg.get('source',{})
    collections=[];audit={};dev_sources={}
    for name,manifest in zip(args.source_name,args.source_manifest):
        ds=load_reid(name,manifest=manifest)
        permitted=[s for s in ds.train if args.exclude_camera is None or s.cam!=args.exclude_camera]
        train,dev=heldout_ids(permitted,seed=args.seed)
        if {x.pid for x in train}&{x.pid for x in dev}:raise ValueError('Source dev ID leakage')
        collections.append(train)
        dev_sources[name]=dev
        audit[name]={'train_images':len(train),'dev_images':len(dev),'source_cameras':sorted({s.cam for s in permitted})}
        if args.exclude_camera in audit[name]['source_cameras']:raise ValueError('Held-out camera leakage')
    examples,numclasses=relabel_sources(collections)
    encoder=build_encoder(m['backbone'],pretrained=m.get('pretrained',False),
             num_domain_tokens=m['domain_tokens'],height=m['height'],width=m['width'])
    if args.source_weights:
        state=torch.load(args.source_weights,map_location='cpu',weights_only=True)
        encoder.load_state_dict(state['encoder'] if 'encoder' in state else state,strict=True)
    head=IdentityHead(encoder.embed_dim,numclasses)
    styles=None
    if args.source_style_pool:
        supplied=torch.load(args.source_style_pool,map_location='cpu',weights_only=True)
        styles=supplied['styles'] if isinstance(supplied,dict) else supplied
        if styles.ndim!=2 or styles.size(1)!=6*encoder.embed_dim and m['backbone'] in ('tiny','vit_b_16','deit_s'):
            raise ValueError('Source-style pool shape does not match encoder 6d statistics')
    trainer=SourceTrainer(encoder,head,c,device=args.device,style_pool=styles)
    epochs=args.epochs or c.get('epochs',120)
    trainer.setup_schedule(epochs)
    P=c.get('identities_per_batch',16);K=c.get('instances_per_id',4)
    batches=loader(examples,batch=P*K,train=True,height=m['height'],width=m['width'],P=P,K=K,seed=args.seed)
    out=Path(args.save_dir);out.mkdir(parents=True,exist_ok=True)
    metadata={'source_names':args.source_name,'outer_target':args.outer_target,
              'excluded_camera':args.exclude_camera,'source_audit':audit,'seed':args.seed,
              'checkpoint_selection':'source_only_identity_disjoint_dev_mAP',
              'style_pool_origin':'user_source_only' if args.source_style_pool else 'current_source_batch',
              'config':cfg}
    write_json(out/'source_audit.json',metadata)
    best=-1.0;best_training_loss=float('inf');hist=[]
    for epoch in range(1,epochs+1):
        batches.sampler.set_epoch(epoch)
        row=trainer.epoch(batches)
        row['epoch']=epoch
        val_map,by_domain=source_dev_map(encoder,dev_sources,m,args.device)
        row['source_dev_map']=val_map
        row['source_dev_by_domain']=by_domain
        hist.append(row)
        save_best=False
        if val_map is not None and val_map>best:
            best=val_map
            metadata['checkpoint_selection']='source_only_identity_disjoint_dev_mAP'
            save_best=True
        elif val_map is None and best<0 and row['loss']<best_training_loss:
            # With fewer than two eligible source cameras, source-only cross-
            # camera validation is undefined. A source-loss-only checkpoint is
            # permitted but MUST NOT be described as mAP-selected.
            best_training_loss=row['loss']
            metadata['checkpoint_selection']='source_training_loss_fallback_no_crosscamera_dev'
            save_best=True
        if save_best:
            torch.save({'encoder':encoder.state_dict(),'identity_head':head.state_dict(),
                        'metadata':metadata,'epoch':epoch},out/'source_best.pt')
        print(f'Epoch {epoch}/{epochs}: {row}',flush=True)
    torch.save({'encoder':encoder.state_dict(),'identity_head':head.state_dict(),
                'metadata':metadata,'epoch':epochs},out/'source_final.pt')
    write_json(out/'history.json',hist)

if __name__=='__main__':main()
