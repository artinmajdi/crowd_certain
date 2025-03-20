import json
import logging
import multiprocessing
import os
import pathlib

import pandas as pd

from crowd_certain.utilities.parameters import params
from crowd_certain.utilities.parameters.settings import ConfigManager
from crowd_certain.utilities.visualizer import Aim1_3_Data_Analysis_Results

config = ConfigManager.get_settings()


multiprocessing.set_start_method('spawn', force=True)
logging.basicConfig(level=logging.DEBUG)

aim1_3 = Aim1_3_Data_Analysis_Results(config=config).update()


aim1_3.figure_metrics_mean_over_seeds_per_dataset_per_worker(metric=params.EvaluationMetricNames.ACC, nl=3, figsize=(12,10), font_scale=1.8)


