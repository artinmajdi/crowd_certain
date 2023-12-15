from crowd_certain.utilities import utils
from crowd_certain.utilities.settings import get_settings
from crowd_certain.utilities.params import DatasetNames
import cProfile
import pstats

def main():

	config = get_settings()

	aim1_3 = utils.Aim1_3_Data_Analysis_Results(config=config)

	# aim1_3.figure_F_heatmap( metric_name='F_eval_one_dataset_all_labelers', dataset_name=DatasetNames.KR_VS_KP, figsize=(13,8), font_scale=2)


if __name__ == '__main__':

	with cProfile.Profile() as pr:
		main()

	p = pstats.Stats(pr)

	# this will sort and print the top 10 functions by cumulative time
	# p.sort_stats('cumulative').print_stats(20)

	p.sort_stats(pstats.SortKey.TIME)

	p.dump_stats(filename='needs_profiling.prof')
