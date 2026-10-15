"""Use only three eligible inner source validation domains per outer target."""
import argparse,json
from data.datasets import b_inner
from utils.experiment import write_json

def select_nested(results):
    selected={}
    for target,candidates in results.items():
        allowed=set(b_inner(target))
        means={}
        for key,values in candidates.items():
            if set(values)!=allowed:raise ValueError(f'{target} lacks valid inner-fold results')
            means[key]=sum(values.values())/len(values)
        if not means:raise ValueError('No candidate results')
        best=max(sorted(means),key=means.get)
        selected[target]={'chosen_config':best,'mean_source_validation_map':means[best]}
    return selected

def main():
    p=argparse.ArgumentParser();p.add_argument('--inner-results',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    with open(args.inner_results) as f:data=json.load(f)
    write_json(args.output,select_nested(data))

if __name__=='__main__':main()
