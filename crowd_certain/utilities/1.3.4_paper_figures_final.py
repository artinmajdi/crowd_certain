
from crowd_certain.utilities import utils_refactored_new as utils
from crowd_certain.utilities.settings import get_settings
from crowd_certain.utilities.params import EvaluationMetricNames

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_name', type=str,
                    # default='mushroom', help='dataset name')

# args = parser.parse_args()

# dataset_name = args.dataset_name if hasattr( args, 'dataset_name') else 'mushroom'
# config = utils.reading_user_input_arguments(dataset_mode='read_arff', dataset_name='ionosphere', outputs_mode='load_local', save_outputs=False, parallel_processing=False)  # , nlabelers_min_max=[5,6], **params_for_debugging)
config = get_settings()

aim1_3 = utils.Aim1_3_Data_Analysis_Results(config=config)

# aim1_3.figure_metrics_mean_over_seeds_per_dataset_per_worker(metric=EvaluationMetricNames.ACC, nl=3, figsize=(12,10), font_scale=1.8)

aim1_3.figure_metrics_all_datasets_workers(figsize=(13,15), font_scale=2)

# aim1_3.figure_weight_quality_relation()

# cProfile.run('run()', 'profiler.txt')
# import pstats
# p = pstats.Stats('profiler.txt')
# # this will sort and print the top 10 functions by cumulative time
# p.sort_stats('cumulative').print_stats(20)
