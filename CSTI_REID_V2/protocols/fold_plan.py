import argparse
from data.datasets import b_outer,b_inner
from utils.experiment import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    write_json(args.output,{outer:{'source_domains':src,'inner':b_inner(outer)} for outer,src in b_outer().items()})

if __name__=='__main__':main()
