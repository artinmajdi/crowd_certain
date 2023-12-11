
from crowd_certain.utilities import utils
from crowd_certain.utilities.settings import get_settings
from crowd_certain.utilities.params import EvaluationMetricNames, DatasetNames

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_name', type=str,
                    # default='mushroom', help='dataset name')

# args = parser.earse_argse)

# dataset_name = args.dataset_name if hasattr( args, 'dataset_name') else 'mushroom'
# config = utils.reading_user_input_arguments(dataset_mode='read_arff', dataset_name='ionosphere', outputs_mode='load_local', save_outputs=False, parallel_processing=False)  # , nlabelers_min_max=[5,6], **params_for_debugging)
config = get_settings()

aim1_3 = utils.Aim1_3_Data_Analysis_Results(config=config)

aim1_3.figure_F_heatmap( metric_name='F_eval_one_dataset_all_labelers', dataset_name=DatasetNames.KR_VS_KP, figsize=(13,8), font_scale=2)



# cProfile.run('run()', 'profiler.txt')
# import pstats
# p = pstats.Stats('profiler.txt')
# # this will sort and print the top 10 functions by cumulative time
# p.sort_stats('cumulative').print_stats(20)
