from __future__ import print_function
from utils.common import *
from utils.pipeline_impact import pipeline_impact
import warnings
warnings.filterwarnings("ignore")


def main():
    args = parse_args()
    pipeline_impact(args.toy_example)


main()
