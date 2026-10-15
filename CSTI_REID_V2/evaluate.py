"""Source-free online evaluation, strict camera and domain isolation."""
import argparse
from pathlib import Path
import torch,numpy as np
from data.datasets import load_reid,loader,c_fold,b_outer
from models.encoder import build_encoder
from models.csti_reid import CSTIAdapter
from utils.experiment import load_config,read_state,set_seed,write_json
from utils.metrics import score,camera_macro
from utils.stats import summary

def load_source(path,cfg,target,protocol,camera=None,allow_unaudited=False):
    obj=read_state(path)
    meta=obj.get('metadata',{})
    if not allow_unaudited:
        if not meta:raise ValueError('Checkpoint has no source-only audit metadata')
        if protocol=='A' and target in meta.get('source_names',[]):
            raise ValueError('Protocol A cannot evaluate its labeled source as a held-out target')
        if protocol=='B':
            expected=set(b_outer()[target])
            if (meta.get('outer_target')!=target or
                set(meta.get('source_names',[]))!=expected):
                raise ValueError('Protocol B requires exactly the three permitted source domains')
        if protocol=='C':
            if meta.get('excluded_camera')!=camera or target not in meta.get('source_names',[]):
                raise ValueError('Camera fold must use a source checkpoint excluding that camera')
            sourcecams=meta.get('source_audit',{}).get(target,{}).get('source_cameras',[])
            if camera in sourcecams:raise ValueError('Held-out camera included during source training')
    m=cfg['model']
    model=build_encoder(m['backbone'],pretrained=False,num_domain_tokens=m['domain_tokens'],
                        height=m['height'],width=m['width'])
    model.load_state_dict(obj['encoder'] if 'encoder' in obj else obj,strict=True)
    return model.eval()

def evaluate_fold(ds,queries,gallery,checkpoint,cfg,seed,device,variant,protocol,camera,allow_unaudited):
    set_seed(seed)
    encoder=load_source(checkpoint,cfg,ds.name,protocol,camera,allow_unaudited)
    opt={**cfg['online'],'seed':seed}
    adapter=CSTIAdapter(encoder,opt,device,variant)
    m=cfg['model']
    gl=loader(gallery,batch=opt['batch_size'],height=m['height'],width=m['width'])
    gids,gcams=adapter.cache_gallery(gl)
    ql=loader(queries,batch=opt['batch_size'],height=m['height'],width=m['width'])
    sim=[];qids=[];qc=[]
    for images,pids,cams in ql:
        pred=adapter.step(images)
        # Pre-update query embeddings and fixed source gallery; no online labels in adapter.
        sim.append((pred['features'].to(device)@adapter.gallery.T).cpu().numpy())
        qids.extend([int(x) for x in pids]);qc.extend([int(x) for x in cams])
    whole=np.concatenate(sim)
    result=score(whole,qids,qc,gids,gcams)
    # Post-hoc window diagnostics only. Target IDs never enter the adapter.
    win=int(opt.get('diagnostic_window',128))
    windows=[]
    for start in range(0,len(qids),win):
        stop=min(start+win,len(qids))
        try:
            m=score(whole[start:stop],qids[start:stop],qc[start:stop],gids,gcams)
            windows.append({'first_query':start,'last_query_exclusive':stop,'mAP':m['mAP']})
        except ValueError:
            continue
    result['diagnostic_windows']=windows
    if len(windows)>=2:
        from scipy.stats import kendalltau
        vals=[w['mAP'] for w in windows]
        tau=kendalltau(range(len(vals)),vals).statistic
        result['endpoint_map_change']=float(vals[-1]-vals[0])
        result['mann_kendall_tau']=float(tau) if np.isfinite(tau) else None
        result['window_map_variance']=float(np.var(vals,ddof=1))
    return result,adapter.trace

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/paper.yaml')
    p.add_argument('--protocol',choices=['A','B','C'],required=True)
    p.add_argument('--target',required=True)
    p.add_argument('--target-manifest',required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--checkpoint-pattern')
    p.add_argument('--n-runs',type=int,default=5)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--variant',choices=['full','simple_sa','no_iwbre','no_rncc','no_muc','no_dppm'],default='full')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',required=True)
    p.add_argument('--allow-unaudited-checkpoint',action='store_true')
    args=p.parse_args()
    if args.protocol=='C' and (not args.checkpoint_pattern or '{camera}' not in args.checkpoint_pattern):
        p.error('Protocol C requires --checkpoint-pattern containing {camera}')
    if args.protocol!='C' and not args.checkpoint:p.error('A/B requires --checkpoint')
    cfg=load_config(args.config)
    ds=load_reid(args.target,manifest=args.target_manifest)
    runs=[]
    for seed in range(args.seed,args.seed+args.n_runs):
        if args.protocol=='C':
            folds=[];traces={}
            for c in sorted({s.cam for s in ds.query}):
                try:source,queries,gallery=c_fold(ds,c)
                except ValueError:continue
                checkpoint=args.checkpoint_pattern.format(camera=c)
                if not Path(checkpoint).is_file():raise FileNotFoundError(checkpoint)
                metric,trace=evaluate_fold(ds,queries,gallery,checkpoint,cfg,seed,args.device,
                    args.variant,'C',c,args.allow_unaudited_checkpoint)
                folds.append({'camera':c,**metric});traces[str(c)]=trace
            if not folds:raise ValueError('No eligible camera folds')
            result={**camera_macro(folds),'camera_folds':folds,'trace':traces}
        else:
            metric,trace=evaluate_fold(ds,ds.query,ds.gallery,args.checkpoint,cfg,seed,args.device,
                args.variant,args.protocol,None,args.allow_unaudited_checkpoint)
            result={**metric,'trace':trace}
        result['seed']=seed;runs.append(result)
        print(f"Seed {seed}: mAP={result['mAP']:.4f} Rank1={result['rank1']:.4f}",flush=True)
    agg={k:summary([r[k] for r in runs]) if len(runs)>=2 else {'n':1,'mean':runs[0][k],'std':None}
         for k in ('mAP','rank1')}
    write_json(args.output,{'protocol':args.protocol,'target':args.target,'variant':args.variant,
                            'audited':not args.allow_unaudited_checkpoint,'runs':runs,'summary':agg})
    print(agg)

if __name__=='__main__':main()
