"""NEXUS command line interface."""
import argparse, json
from .generator import generate_data
from .pipeline import run_pipeline
from .modeling import train_models

def main():
    parser=argparse.ArgumentParser(prog="nexus",description="NEXUS financial data platform")
    sub=parser.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate"); g.add_argument("--rows",type=int,default=100000); g.add_argument("--seed",type=int,default=42)
    sub.add_parser("pipeline")
    sub.add_parser("train")
    r=sub.add_parser("run"); r.add_argument("--rows",type=int,default=100000); r.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.command in ("generate","run"): generate_data(args.rows,args.seed)
    if args.command in ("pipeline","run"): result=run_pipeline(); print("Quality:",json.dumps(result["quality"],indent=2))
    if args.command in ("train","run"): print("Model metrics:",json.dumps(train_models(),indent=2))
if __name__=="__main__": main()
