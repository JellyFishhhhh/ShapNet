import argparse
import json
from .runner import run


def main():
    parser=argparse.ArgumentParser(description='Independent TriMorph-MIL patient active learning')
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-updates',type=int,help='Pause after this many additional optimizer updates')
    parser.add_argument('--max-rounds',type=int,help='Pause after this many total completed rounds')
    args=parser.parse_args()
    print(json.dumps(run(args.config,args.resume,args.max_updates,args.max_rounds)))


if __name__=='__main__':
    main()
