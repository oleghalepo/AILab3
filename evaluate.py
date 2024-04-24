from models import evaluate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ground-truth')
parser.add_argument('--predictions')
args = parser.parse_args()
evaluate(args.ground_truth, args.predictions)
