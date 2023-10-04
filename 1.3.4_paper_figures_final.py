
from main.aims.crowd import utils_crowd as utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,
                    default='mushroom', help='dataset name')

args = parser.parse_args()

# dataset_name = args.dataset_name if hasattr( args, 'dataset_name') else 'mushroom'
config = utils.reading_user_input_arguments(dataset_mode='read_arff', dataset_name='ionosphere', outputs_mode='load_local', save_outputs=False, parallel_processing=False)  # , nlabelers_min_max=[5,6], **params_for_debugging)


aim1_3 = utils.Aim1_3_Data_Analysis_Results(config=config)

aim1_3.figure_weight_quality_relation()

# cProfile.run('run()', 'profiler.txt')
# import pstats
# p = pstats.Stats('profiler.txt')
# # this will sort and print the top 10 functions by cumulative time
# p.sort_stats('cumulative').print_stats(20)
