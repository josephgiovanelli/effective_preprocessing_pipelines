from __future__ import print_function
from utils.common import *
from results_processors.pipeline_impact import pipeline_impact
from results_processors.pipeline_construction import pipeline_construction
from results_processors.evaluation import evaluation1, evaluation2, evaluation3
from results_processors.exploratory_analysis import exploratory_analysis
import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger("matplotlib.font_manager").disabled = True


def process_results(args):
    """
    Process the results according to the given parameters

    Args:
        args: taken from utils.common.parse_args
    """
    if args.experiment == "pipeline_impact":
        pipeline_impact(args.toy_example)
    elif args.experiment == "pipeline_construction":
        pipeline_construction(args.toy_example)
    elif args.experiment == "evaluation":
        evaluation1(args.toy_example)
        evaluation2(args.toy_example)
        evaluation3(args.toy_example)
    elif args.experiment == "exploratory_analysis":
        exploratory_analysis(args.toy_example)


if __name__ == "__main__":
    args = parse_args()
    process_results(args)
