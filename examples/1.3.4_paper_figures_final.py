from src.crowd_certain.utilities import utils
from crowd_certain.utilities.parameters.settings import ConfigManager

def main():

	config = ConfigManager.get_settings()

	aim1_3 = utils.Aim1_3_Data_Analysis_Results(config=config).update()

	# aim1_3.figure_F_heatmap( metric_name='F_eval_one_dataset_all_workers', dataset_name=DatasetNames.KR_VS_KP, figsize=(13,8), font_scale=2)


if __name__ == '__main__':

	main()
	print(' sf s  s')
