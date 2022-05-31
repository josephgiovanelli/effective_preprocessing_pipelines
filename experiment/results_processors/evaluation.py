from utils.common import *
from utils.evaluation import evaluation1, evaluation2, evaluation3

args = parse_args()
evaluation1(args.toy_example)
evaluation2(args.toy_example)
evaluation3(args.toy_example)