import argparse
from data.datasets import load_reid,c_fold
from utils.experiment import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    ds=load_reid(a.name,manifest=a.manifest)
    folds={}
    for cam in sorted({s.cam for s in ds.query}):
        try:
            source,queries,gallery=c_fold(ds,cam)
            folds[str(cam)]={'eligible':True,'source_images':len(source),'queries':len(queries)}
        except ValueError as e:folds[str(cam)]={'eligible':False,'reason':str(e)}
    write_json(a.output,{'name':a.name,'camera_folds':folds})

if __name__=='__main__':main()
