import argparse

import os

# Common paths
RESOURCES_PATH = os.path.join("./", "resources")
SCENARIO_PATH = os.path.join(RESOURCES_PATH, "scenarios")
RAW_RESULT_PATH = os.path.join(RESOURCES_PATH, "raw_results")
ARTIFACTS_PATH = os.path.join(RESOURCES_PATH, "artifacts")
META_FEATURES_PATH = os.path.join(RESOURCES_PATH, "meta_features")
DATASETS_PATH = os.path.join(RESOURCES_PATH, "datasets")


algorithms = ['RandomForest', 'NaiveBayes', 'KNearestNeighbors']

# Suite OpenML-CC18
# Benchmark_suite = openml.study.get_suite('OpenML-CC18') # obtain the benchmark suite
benchmark_suite = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307, 
                       300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1510, 1489, 1494, 1497, 1501, 
                       1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381, 40499, 
                       40668, 40966, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978, 
                       40670, 40701]

# Suite AutoML
extended_benchmark_suite = [41145, 41156, 41157, 4541, 41158, 42742, 40498, 42734, 41162, 42733, 42732, 1596, 40981, 40685, 
                        4135, 41142, 41161, 41159, 41163, 41164, 41138, 41143, 41146, 41150, 40900, 41165, 41166, 41168, 41169, 
                        41147, 1111, 1169, 41167, 41144, 1515, 1457, 181]
# Suite all-classification-tasks
large_comparison_classification_tasks = [12, 9, 7, 10, 14, 15, 6, 13, 11, 5, 2, 4, 3, 20, 49, 179, 164, 171, 182, 163, 181, 172, 
                        38, 27, 44, 28, 23, 37, 43, 41, 36, 39, 35, 62, 57, 151, 56, 60, 59, 61, 119, 137, 22, 29, 16, 24, 32, 30, 
                        26, 18, 34, 384, 387, 383, 377, 357, 375, 51, 188, 183, 185, 275, 184, 186, 255, 187, 251, 388, 397, 400, 401, 
                        392, 389, 391, 443, 444, 382, 554, 488, 479, 481, 480, 682, 679, 683, 861, 865, 859, 858, 862, 857, 864, 863, 860, 
                        378, 477, 470, 467, 476, 475, 474, 468, 472, 469, 771, 767, 765, 766, 769, 768, 773, 770, 772, 818, 811, 812, 819, 
                        815, 816, 817, 813, 814, 851, 853, 852, 854, 848, 850, 847, 849, 855, 381, 455, 452, 453, 449, 454, 457, 451, 450, 
                        448, 786, 788, 787, 784, 785, 790, 789, 792, 791, 840, 839, 844, 842, 846, 838, 841, 843, 845, 300, 313, 312, 316, 
                        311, 329, 335, 334, 333, 722, 720, 724, 723, 728, 725, 727, 721, 726, 738, 745, 742, 741, 746, 739, 740, 744, 743, 
                        802, 810, 804, 805, 806, 803, 807, 808, 42, 31, 46, 54, 40, 55, 52, 48, 53, 50, 460, 465, 463, 458, 462, 461, 466, 
                        459, 464, 757, 758, 759, 760, 761, 764, 756, 762, 763, 834, 831, 836, 833, 837, 832, 830, 829, 835, 340, 342, 343, 
                        339, 350, 338, 337, 336, 734, 732, 735, 737, 733, 736, 731, 729, 730, 778, 779, 775, 777, 780, 774, 783, 776, 782, 
                        825, 828, 821, 823, 820, 824, 827, 826, 25, 285, 307, 279, 276, 278, 277, 310, 718, 685, 719, 714, 716, 694, 717, 
                        713, 715, 753, 748, 752, 750, 751, 755, 749, 754, 747, 801, 796, 798, 797, 800, 794, 793, 795, 799, 902, 887, 888, 
                        866, 896, 903, 898, 900, 901, 895, 981, 978, 982, 984, 983, 980, 979, 976, 977, 873, 876, 877, 882, 880, 886, 889, 
                        893, 894, 885, 868, 879, 874, 871, 869, 878, 884, 870, 867, 875, 953, 952, 956, 949, 957, 950, 951, 955, 1107, 1104, 
                        1109, 1102, 1116, 1119, 1115, 1117, 915, 928, 881, 923, 927, 922, 929, 926, 925, 891, 892, 921, 920, 918, 914, 919, 
                        917, 913, 916, 987, 992, 990, 989, 985, 988, 986, 991, 930, 897, 942, 947, 944, 941, 945, 943, 946, 1017, 1018, 1012, 
                        1020, 1013, 1021, 1019, 1015, 1014, 899, 905, 904, 909, 907, 910, 906, 908, 912, 911, 967, 975, 972, 970, 971, 968, 973, 
                        969, 974, 1057, 1065, 1056, 1064, 1063, 1059, 1061, 1062, 1060, 924, 938, 934, 939, 937, 932, 933, 936, 935, 931, 1008, 
                        1004, 1007, 1010, 1009, 1006, 1005, 1011, 1003, 940, 1038, 1041, 1022, 1040, 1026, 1023, 1025, 954, 961, 965, 966, 964, 
                        963, 958, 962, 960, 959, 1045, 1053, 1044, 1050, 1049, 1054, 1046, 1055, 1048, 993, 1002, 1001, 998, 1000, 995, 999, 994, 
                        996, 997, 1101, 1066, 1071, 1069, 1068, 1067, 1075, 1073, 1120, 1121]

# Dataset used to verify the impact of pre-processing
pipeline_impact_suite = [1461]

def parse_args():
    """Parse the arguments given via CLI.

    Returns:
        dict: arguments and their values.
    """
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=False, help="step of the pipeline to execute")
    parser.add_argument("-exp", "--experiment", nargs="?", type=str, required=False, help="type of the experiments")
    parser.add_argument("-mode", "--mode", nargs="?", type=str, required=False, help="algorithm or algorithm_pipeline")
    parser.add_argument("-toy", "--toy_example", action='store_true', default=False, help="wether it is a toy example or not")
    parser.add_argument("-cache", "--cache", action='store_true', default=False, help="wether to use the intermediate results or not")
    args = parser.parse_args()
    return args

def create_directory(result_path, directory):
    """Create a directory in the specified path.

    Args:
        result_path: where to create a directory.
        directory: name of the directory.

    Returns:
        os.path: the resulting path.
    """
    result_path = os.path.join(result_path, directory)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path

def get_filtered_datasets(experiment, toy):
    """Retrieve the dataset list for a certain experiment.

    Args:
        experiment: keyword of the experiment.
        toy: whether it is the toy example or not.

    Returns:
        list: list of OpenML ids.
    """
    if experiment == "pipeline_impact":
        return pipeline_impact_suite
    else:
        import pandas as pd
        df = pd.read_csv(os.path.join(META_FEATURES_PATH, "simple-meta-features.csv"))
        df = df.loc[df['did'].isin(list(dict.fromkeys(benchmark_suite + extended_benchmark_suite + [10, 20, 26])))]
        df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
        df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
        df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
        if toy:
            df = df.loc[df['NumberOfInstances'] <= 2000]
            df = df.loc[df['NumberOfFeatures'] <= 10]
            df = df.sort_values(by=['NumberOfInstances', 'NumberOfFeatures'])
            df = df[:10]
        df = df['did']
        return df.values.flatten().tolist()