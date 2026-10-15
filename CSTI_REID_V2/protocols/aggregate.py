import argparse,json
from utils.stats import summary,paired
from utils.experiment import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--ours',required=True);p.add_argument('--baseline')
    p.add_argument('--output',required=True);a=p.parse_args()
    with open(a.ours) as f:ours=json.load(f)
    x={r['seed']:r['mAP'] for r in ours['runs']};out={'ours':summary(list(x.values()))}
    if a.baseline:
        with open(a.baseline) as f:base=json.load(f)
        if (ours['protocol'],ours['target'])!=(base['protocol'],base['target']):
            raise ValueError('Comparison requires identical target and protocol')
        y={r['seed']:r['mAP'] for r in base['runs']}
        out.update(baseline=summary(list(y.values())),paired=paired(x,y))
    write_json(a.output,out)

if __name__=='__main__':main()
