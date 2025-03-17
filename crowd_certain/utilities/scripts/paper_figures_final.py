# Original file: /Users/artinmajdi/Documents/GitHubs/PhD/code/submodules/crowd/crowd_certain/utilities/1.3.4_paper_figures_final.py
from crowd_certain.utilities import Aim1_3_Data_Analysis_Results, get_settings
from crowd_certain.utilities.params import DatasetNames
import cProfile
import pstats

def main():
    config = get_settings()
    aim1_3 = Aim1_3_Data_Analysis_Results(config=config).update()
    # aim1_3.figure_F_heatmap(metric_name='F_eval_one_dataset_all_workers', dataset_name=DatasetNames.KR_VS_KP, figsize=(13,8), font_scale=2)

if __name__ == '__main__':
    # with cProfile.Profile() as pr:
    #     main()
    # p = pstats.Stats(pr)
    # p.sort_stats(pstats.SortKey.TIME)
    # p.dump_stats(filename='needs_profiling.prof')

    main()
    print('Completed successfully')
