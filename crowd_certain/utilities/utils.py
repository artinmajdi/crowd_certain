import argparse
import json
import multiprocessing
import os
import pathlib
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import *
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
from crowdkit import aggregation as crowdkit_aggregation
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.special import bdtrc
from sklearn import ensemble as sk_ensemble, metrics as sk_metrics
from tqdm import tqdm_notebook as tqdm

from main.aims.crowd import load_data_crowd as load_data


class Dict2Class:
	""" It takes a dictionary and turns it into a class """
	def __init__(self, my_dict):
		for key in my_dict:
			setattr(self, key, my_dict[key])


def func_callBacks(dir_save='', mode='min', monitor='val_loss'):
	checkPointer = tf.keras.callbacks.ModelCheckpoint(filepath=dir_save, monitor=monitor, verbose=1, save_best_only=True, mode=mode)

	# Reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1, min_delta=0.005 , patience=10, verbose=1, save_best_only=True, mode=mode , min_lr=0.9e-5 , )

	# CSVLogger = tf.keras.callbacks.CSVLogger(dir_save + '/results.csv', separator=',', append=False)

	# earlyStopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, verbose=1, mode=mode, restore_best_weights=True)

	return [checkPointer]  # [ earlyStopping , CSVLogger]


def reading_user_input_arguments(argv=None, jupyter=True, config_name='config_crowd.json', mlflow_setup=None, **kwargs) -> argparse.Namespace:

	def parse_args() -> argparse.Namespace:
		"""	Getting the arguments from the command line
			Problem: 	Jupyter Notebook automatically passes some command-line arguments to the kernel.
						When we run argparse.ArgumentParser.parse_args(), it tries to parse those arguments, which are not recognized by your argument parser.
			Solution: 	To avoid this issue, you can modify your get_args() function to accept an optional list of command-line arguments, instead of always using sys.argv.
						When this list is provided, the function will parse the arguments from it instead of the command-line arguments. """

		# If argv is not provided, use sys.argv[1:] to skip the script name
		args = [] if jupyter else (argv or sys.argv[1:])

		args_list = [
					# Dataset
					dict(name='dataset_name', type=str, help='Name of the dataset'),
					dict(name='data_mode', type=str, help='Dataset mode: train or valid'),

					# Model
					dict(name='model_name'   , type=str , help='Name of the pre_trained model.' ),
					dict(name='architecture' , type=str , help='Name of the architecture'       ),

					# Training
					dict(name = 'batch_size'     , type = int   , help = 'Number of batches to process' ),
					dict(name = 'n_epochs'       , type = int   , help = 'Number of epochs to process'  ),
					dict(name = 'learning_rate'  , type = float , help = 'Learning rate'                ),
					dict(name = 'n_augmentation' , type = int   , help = 'Number of augmentations'      ),

					# Hyperparameter Optimization
					dict(name = 'parent_condition_mode', type = str, help = 'Parent condition mode: truth or predicted' ),
					dict(name = 'approach'             , type = str, help = 'Hyper parameter optimization approach' ),
					dict(name = 'max_evals'            , type = int, help = 'Number of evaluations for hyper parameter optimization' ),
					dict(name = 'n_batches_to_process' , type = int, help = 'Number of batches to process' ),

					# MLFlow
					dict(name='RUN_MLFLOW'            , type=bool  , help='Run MLFlow'                                             ),
					dict(name='KILL_MLFlow_at_END'    , type=bool  , help='Kill MLFlow'                                            ),

					# Config
					dict(name='config'                , type=str   , help='Path to config file' , default='config.json'             ),
					]

		# Initializing the parser
		parser = argparse.ArgumentParser()

		# Adding arguments
		for g in args_list:
			parser.add_argument(f'--{g["name"].replace("_","-")}', type=g['type'], help=g['help'], default=g.get('default')) # type: ignore

		# Filter out any arguments starting with '-f'
		filtered_argv = [arg for arg in args if not (arg.startswith('-f') or 'jupyter/runtime' in arg.lower())]

		# Parsing the arguments
		return parser.parse_args(args=filtered_argv)

	def updating_config_with_kwargs(updated_args):
		if kwargs and len(kwargs) > 0:
			for key in kwargs.keys():
				updated_args[key] = kwargs[key]
		return updated_args

	def get_config(args): # type: (argparse.Namespace) -> argparse.Namespace

		def add_params_from_mlflow():
			if (args.params_source == 'load_MLFlow') and mlflow_setup:
				mlflow_params = eval(mlflow_setup.run.data.params)
				for name in ['num_seeds', 'num_simulations', 'low_dis', 'high_dis']:
					setattr(args, name, mlflow_params[name])
				args.config.workers_list = mlflow_params['workers_list']

		# Loading the config.json file
		config_dir =  os.path.join(os.path.dirname(__file__), config_name if jupyter else args.config)

		if os.path.exists(config_dir):
			with open(config_dir) as f:
				config_raw = json.load(f)

			# converting args to dictionary
			args_dict = vars(args) if args else {}

			# Updating the config with the arguments as command line input
			updated_args ={key: args_dict.get(key) or values for key, values in config_raw.items() }

			# Updating the config with the arguments as function input: used for facilitating the jupyter notebook access
			updated_args = updating_config_with_kwargs( updated_args )

			# Convert the dictionary to a Namespace
			args = argparse.Namespace(**updated_args)

			# Updating the paths to their absolute path
			args.outputs_path 				= pathlib.Path(__file__).parent.joinpath(args.outputs_path)
			args.dataset_path 			    = pathlib.Path(__file__).parent.joinpath(args.dataset_path)
			args.outputs_path 			    = pathlib.Path(__file__).parent.joinpath(args.outputs_path)
			args.MLFlow_run_name 			= f'{args.dataset_name}-{args.model_name}'
			args.workers_list 			= list(range(*args.nlabelers_min_max))


		add_params_from_mlflow()

		return args

	# Updating the config file
	return  get_config(args=parse_args())

class LoadSaveFile:
	def __init__(self, path):
		self.path = path

	def load(self, index_col=None, header=None):

		if self.path.exists():

			if self.path.suffix == '.pkl':
				with open(self.path, 'rb') as f:
					return pickle.load(f)

			elif self.path.suffix == '.csv':
				return pd.read_csv(self.path)

			elif self.path.suffix == '.xlsx':
				return pd.read_excel(self.path, index_col=index_col, header=header)

		return None

	def dump(self, file, index=False, upload_artifact=False, artifact_path=''):

		self.path.parent.mkdir(parents=True, exist_ok=True)

		if self.path.suffix == '.pkl':
			with open(self.path, 'wb') as f:
				pickle.dump(file, f)

		elif self.path.suffix == '.csv':
			file.to_csv(self.path, index=index)

		elif self.path.suffix == '.xlsx':
			file.to_excel(self.path, index=index)

		if upload_artifact:
			mlflow.log_artifact(local_path=self.path, artifact_path=artifact_path)


""" Model training and validation """
def get_architecture(architecture='DenseNet121', input_shape=[224, 224, 3], num_classes=14, activation='sigmoid', first_index_trainable=None, weights='imagenet'):

	def custom_model(input_tensor):
		modelc = tf.keras.layers.Conv2D(4, kernel_size=(3, 3), activation='relu')(input_tensor)
		modelc = tf.keras.layers.BatchNormalization()(modelc)
		modelc = tf.keras.layers.MaxPooling2D(2, 2)(modelc)

		modelc = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu')(modelc)
		modelc = tf.keras.layers.BatchNormalization()(modelc)
		modelc = tf.keras.layers.MaxPooling2D(2, 2)(modelc)

		modelc = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(modelc)
		modelc = tf.keras.layers.BatchNormalization()(modelc)
		modelc = tf.keras.layers.MaxPooling2D(2, 2)(modelc)

		modelc = tf.keras.layers.Flatten()(modelc)
		modelc = tf.keras.layers.Dense(32, activation='relu')(modelc)
		modelc = tf.keras.layers.Dense(num_classes, activation='softmax')(modelc)

		return tf.keras.models.Model(inputs=modelc.input, outputs=[modelc])

	input_tensor = tf.keras.layers.Input(input_shape)

	if architecture == 'custom':
		return custom_model(input_tensor)


	pooling            = 'avg'
	include_top        = False
	model_architecture = tf.keras.applications.DenseNet121 # The default architecture

	if   architecture == 'xception'         : model_architecture = tf.keras.applications.Xception
	elif architecture == 'VGG16'            : model_architecture = tf.keras.applications.VGG16
	elif architecture == 'VGG19'            : model_architecture = tf.keras.applications.VGG19
	elif architecture == 'ResNet50'         : model_architecture = tf.keras.applications.ResNet50
	elif architecture == 'ResNet50V2'       : model_architecture = tf.keras.applications.ResNet50V2
	elif architecture == 'ResNet101'        : model_architecture = tf.keras.applications.ResNet101
	elif architecture == 'ResNet101V2'      : model_architecture = tf.keras.applications.ResNet101V2
	elif architecture == 'ResNet152'        : model_architecture = tf.keras.applications.ResNet152
	elif architecture == 'ResNet152V2'      : model_architecture = tf.keras.applications.ResNet152V2
	elif architecture == 'InceptionV3'      : model_architecture = tf.keras.applications.InceptionV3
	elif architecture == 'InceptionResNetV2': model_architecture = tf.keras.applications.InceptionResNetV2
	elif architecture == 'MobileNet'        : model_architecture = tf.keras.applications.MobileNet
	elif architecture == 'MobileNetV2'      : model_architecture = tf.keras.applications.MobileNetV2
	elif architecture == 'DenseNet121'      : model_architecture = tf.keras.applications.DenseNet121
	elif architecture == 'DenseNet169'      : model_architecture = tf.keras.applications.DenseNet169
	elif architecture == 'DenseNet201'      : model_architecture = tf.keras.applications.DenseNet201

	elif int(list(tf.keras.__version__)[2]) >= 4:

		if   architecture == 'EfficientNetB0': model_architecture = tf.keras.applications.EfficientNetB0
		elif architecture == 'EfficientNetB1': model_architecture = tf.keras.applications.EfficientNetB1
		elif architecture == 'EfficientNetB2': model_architecture = tf.keras.applications.EfficientNetB2
		elif architecture == 'EfficientNetB3': model_architecture = tf.keras.applications.EfficientNetB3
		elif architecture == 'EfficientNetB4': model_architecture = tf.keras.applications.EfficientNetB4
		elif architecture == 'EfficientNetB5': model_architecture = tf.keras.applications.EfficientNetB5
		elif architecture == 'EfficientNetB6': model_architecture = tf.keras.applications.EfficientNetB6
		elif architecture == 'EfficientNetB7': model_architecture = tf.keras.applications.EfficientNetB7

	model = model_architecture(weights=weights, include_top=include_top, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling)  # ,classes=num_classes

	assert (first_index_trainable < 0) or (first_index_trainable is None), 'first_index_trainable must be negative'

	if first_index_trainable:
		for layer in model.layers[:first_index_trainable]:
			layer.trainable = False

		for layer in model.layers[first_index_trainable:]:
			layer.trainable = True

	KK = tf.keras.layers.Dense(num_classes, activation=activation, name='predictions')(model.output)

	return tf.keras.models.Model(inputs=model.input, outputs=KK)


def weighted_bce_loss(W):
	def func_loss(y_true, y_pred):
		NUM_CLASSES = y_pred.shape[1]

		loss = 0

		for d in range(NUM_CLASSES):
			y_true = tf.cast(y_true, tf.float32)

			# mask   = tf.keras.backend.cast( tf.keras.backend.not_equal(y_true[:,d], -5),
			#                                 tf.keras.backend.floatx() )

			# loss  += W[d]*tf.keras.losses.binary_crossentropy( y_true[:,d] * mask,
			#                                                    y_pred[:,d] * mask )

			loss += W[d] * tf.keras.losses.binary_crossentropy(y_true[:, d], y_pred[:, d])  # type: ignore

		return tf.divide(loss, tf.cast(NUM_CLASSES, tf.float32))

	return func_loss


def optimize(dir_save, data_loader, epochs, architecture='DenseNet121', activation='sigmoid', first_index_trainable=None, use_multiprocessing=True, model_metrics=[tf.keras.metrics.binary_accuracy], weights='imagenet', num_classes=0):

	# architecture
	model = get_architecture( architecture     = architecture,
							  input_shape           = list(data_loader.target_size) + [3],
							  num_classes           = num_classes,
							  activation            = activation,
							  first_index_trainable = first_index_trainable,
							  weights               = weights )

	model.compile(  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
					loss      = weighted_bce_loss(data_loader.class_weights),
					metrics   = model_metrics )

	# optimization
	callbacks = func_callBacks( dir_save = dir_save + 'best_model.h5',
								mode     = 'min',
								monitor  = 'val_loss')

	history = model.fit(data_loader.generators['train_with_augments'],
						validation_data     = data_loader.generators['valid'],
						epochs              = epochs,
						steps_per_epoch     = data_loader.steps_per_epoch['train'],
						validation_steps    = data_loader.steps_per_epoch['valid'],
						verbose             = 1,
						use_multiprocessing = use_multiprocessing,
						callbacks           = callbacks)

	# saving the optimized model
	model.save( dir_save + 'model.h5',
				overwrite         = True,
				include_optimizer = False)

	return model, history


""" Aim1.1: Taxonomy-based Loss Function """


def measure_loss_acc_on_test_data(data, model, labels):
	NUM_CLASSES = len(labels)

	# Chest dataset valid and test were generator while CIFAR360 is a tuple
	data_type_generator = not isinstance(data, (tuple, list))

	if data_type_generator:
		data.reset()

	L = len(data.filenames) if data_type_generator else data[0].shape[0]

	score_values = {}
	for j in tqdm(range(L)):

		# Chest-xray dataset
		if data_type_generator:
			x_test, y_test = next(data)
			full_path, x, y = data.filenames[j], x_test[0, ...], y_test[0, ...]
			x, y = x[np.newaxis, :], y[np.newaxis, :]

		# CIFAR100 dataset
		else:
			full_path, x, y = f'{j}', data[0][j:j + 1, :], data[1][j:j + 1, :]

		# Estimating the loss & accuracy for instance
		eval_results = model.evaluate(x=x, y=y, verbose=0, return_dict=True)

		# predicting the labels for instance
		pred = model.predict(x=x, verbose=0)

		# Measuring the loss for each class
		loss_per_class = [tf.keras.losses.binary_crossentropy(y[..., d], pred[..., d]) for d in range(NUM_CLASSES)]

		# saving all the infos
		score_values[full_path] = {'full_path': full_path, 'loss_avg': eval_results['loss'],
								   'acc_avg': eval_results['binary_accuracy'],
								   'pred': pred[0], 'pred_binary': pred[0] > 0.5, 'truth': y[0] > 0.5,
								   'loss': np.array(loss_per_class), 'label_names': labels}

	# converting the outputs into panda dataframe
	df = pd.DataFrame.from_dict(score_values).T

	# resetting the index to integers
	df.reset_index(inplace=True)

	# # dropping the old index column
	df = df.drop(['index'], axis=1)

	return df


def measure_mean_accruacy_chexpert(truth, prediction, how_to_treat_nans):
	""" prediction & truth: num_samples thresh_technique num_classes """

	pred_classes = prediction > 0.5

	# truth_nan_applied = self._truth_with_nan_applied()
	truth_nan_applied = apply_nan_back_to_truth(truth=truth, how_to_treat_nans=how_to_treat_nans)

	# measuring the binary truth labels (the nan samples will be fixed below)
	truth_binary = truth_nan_applied > 0.5

	truth_pred_compare = (pred_classes == truth_binary).astype(float)

	# replacing the nan samples back to their nan value
	truth_pred_compare[np.where(np.isnan(truth_nan_applied))] = np.nan

	# measuring teh average accuracy over all samples after ignoring the nan samples
	accuracy = np.nanmean(truth_pred_compare, axis=0) * 100

	# this is for safety measure; in case one of the classes overall accuracy was also nan. if removed, then the integer format below will change to very long floats
	accuracy[np.isnan(accuracy)] = 0
	accuracy = (accuracy * 10).astype(int) / 10

	return accuracy


def apply_nan_back_to_truth(truth, how_to_treat_nans):
	# changing teh samples with uncertain truth label to nan
	truth[truth == -10] = np.nan

	# how to treat the nan labels in the original dataset before measuring the average accuracy
	if how_to_treat_nans == 'ignore': truth[truth == -5] = np.nan
	elif how_to_treat_nans == 'pos': truth[truth == -5] = 1
	elif how_to_treat_nans == 'neg': truth[truth == -5] = 0
	return truth


def measure_mean_uncertainty_chexpert(truth, uncertainty, how_to_treat_nans='ignore'):  # type: (np.ndarray, np.ndarray, str) -> np.ndarray
	""" uncertainty & truth:  num_samples thresh_technique num_classes """

	# adding the nan values back to arrays
	truth_nan_applied = apply_nan_back_to_truth(truth, how_to_treat_nans)

	# replacing the nan samples back to their nan value
	uncertainty[np.where(np.isnan(truth_nan_applied))] = np.nan

	# measuring teh average accuracy over all samples after ignoring the nan samples
	uncertainty_mean = np.nanmean(uncertainty, axis=0)

	# this is for safety measure; in case one of the classes overall accuracy was also nan. if removed, then the integer format below will change to very long floats
	uncertainty_mean[np.isnan(uncertainty_mean)] = 0
	uncertainty_mean = (uncertainty_mean * 1000).astype(int) / 1000

	return uncertainty_mean


""" Below is also part of AIM1.1 and should be corrected and merged into the above """


class Measure_Accuracy_Aim1_2:

	def __init__(self, model, generator, predict_accuracy_mode=False, how_to_treat_nans='ignore',
				 uncertainty_type='std'):  # type: (tf.keras.models.Model.dtype, tf.keras.preprocessing.image.ImageDataGenerator, bool, str, str) -> None
		"""
		how_to_treat_nans:
			ignore: ignoring the nan samples when measuring the average accuracy
			pos: if integer number, it'll treat as postitive
			neg: if integer number, it'll treat as negative """

		self.uncertainty_final = np.array([])
		self.accuracy_final = np.array([])
		self.probs_std_2d = np.array([])
		self.probs_avg_2d = np.array([])
		self.accuracy_all_augs_3d = np.array([])
		self.probs_all_augs_3d = np.array([])
		self.predict_accuracy_mode = predict_accuracy_mode
		self.how_to_treat_nans = how_to_treat_nans
		self.generator = generator
		self.model = model
		self.uncertainty_type = uncertainty_type
		self.truth = np.array([])

		self._setting_params()

	def _setting_params(self):

		self.full_data_length, self.num_classes = self.generator.labels.shape
		self.batch_size           = self.generator.batch_size
		self.number_batches = int(np.ceil(self.full_data_length / self.batch_size))
		self.truth                    = self.generator.labels.astype(float)

	def loop_over_whole_dataset(self):
		"""Looping over all batches """

		probs = np.zeros(self.generator.labels.shape)
		accuracy = None

		# Keras_backend.clear_session()
		self.generator.reset()
		np.random.seed(1)

		for batch_index in tqdm(range(self.number_batches), disable=False):
			# extracting the indexes for batch "batch_index"
			self.generator.batch_index = batch_index
			indexes = next(self.generator.index_generator)

			# print('   extracting data -------')
			self.generator.batch_index = batch_index
			x, _ = next(self.generator)

			# print('   predicting the labels -------')
			probs[indexes, :] = self.model.predict(x, verbose=0)

		# Measuring the accuracy over whole augmented dataset
		if self.predict_accuracy_mode:
			accuracy = measure_mean_accruacy_chexpert(truth=self.truth.copy(), prediction=probs.copy(),
													  how_to_treat_nans=self.how_to_treat_nans)

		return probs, accuracy

	def loop_over_all_augmentations(self, n_augmentation: int = 0):

		self.probs_all_augs_3d = np.zeros((1 + n_augmentation, self.full_data_length, self.num_classes))
		self.accuracy_all_augs_3d = np.zeros((1 + n_augmentation, self.num_classes))

		# Looping over all augmentation scenarios
		for ix_aug in range(n_augmentation):
			print(f'augmentation {ix_aug}/{n_augmentation}')
			probs, accuracy = self.loop_over_whole_dataset()

			self.probs_all_augs_3d[ix_aug, ...] = probs
			self.accuracy_all_augs_3d[ix_aug, ...] = accuracy

		# measuring the average probability over all augmented data
		self.probs_avg_2d = np.mean(self.probs_all_augs_3d, axis=0)

		if self.uncertainty_type == 'std':
			self.probs_std_2d = np.std(self.probs_all_augs_3d, axis=0)

		# Measuring the accruacy for new estimated probability for each sample over all augmented data

		# self.accuracy_final    = self._measure_mean_accruacy(self.probs_avg_2d)
		# self.uncertainty_final = self._measure_mean_std(self.probs_std_2d)

		self.accuracy_final = measure_mean_accruacy_chexpert(truth=self.truth.copy(),
															 prediction=self.probs_avg_2d.copy(),
															 how_to_treat_nans=self.how_to_treat_nans)
		self.uncertainty_final = measure_mean_uncertainty_chexpert(truth=self.truth.copy(),
																   uncertainty=self.probs_std_2d.copy(),
																   how_to_treat_nans=self.how_to_treat_nans)


def apply_technique_aim_1_2(data_generator, data_generator_aug, how_to_treat_nans='ignore', model='', n_augmentation=3, uncertainty_type='std'):
	print('running the evaluation on original non-augmented data')

	MA = Measure_Accuracy_Aim1_2(predict_accuracy_mode=True,
								 generator=data_generator,
								 model=model,
								 how_to_treat_nans=how_to_treat_nans,
								 uncertainty_type=uncertainty_type)

	probs_2d_orig, old_accuracy = MA.loop_over_whole_dataset()

	print(' running the evaluation on augmented data including the uncertainty measurement')

	MA = Measure_Accuracy_Aim1_2(predict_accuracy_mode=True,
								 generator=data_generator_aug,
								 model=model,
								 how_to_treat_nans=how_to_treat_nans,
								 uncertainty_type=uncertainty_type)

	MA.loop_over_all_augmentations(n_augmentation=n_augmentation)

	final_results = {'old-accuracy': old_accuracy,
					 'new-accuracy': MA.accuracy_final,
					 'std': MA.uncertainty_final}

	return probs_2d_orig, final_results, MA


def estimate_maximum_and_change(all_accuracies, label_names=None):  # type: (np.ndarray, list) -> pd.DataFrame

	if label_names is None: label_names = []

	columns = ['old-accuracy', 'new-accuracy', 'std']

	# creating a dataframe from accuracies
	df = pd.DataFrame(all_accuracies, index=label_names)

	# adding the 'maximum' & 'change' columns
	df['maximum'] = df.columns[df.values.argmax(axis=1)]
	df['change'] = df[columns[1:]].max(axis=1) - df[columns[0]]

	# replacing "0" values to "--" for readability
	df.maximum[df.change == 0.0] = '--'
	df.change[df.change == 0.0] = '--'

	return df


""" Aim1.3: Soft-weighted MV """


class Aim1_3_ApplyingBenchmarksToCrowdData:

	BENCHMARKS = ['KOS', 'MajorityVote', 'MMSR', 'Wawa', 'ZeroBasedSkill', 'GLAD', 'DawidSkene']

	def __init__(self, crowd_labels, ground_truth): # type: (Dict, Dict) -> None
		""" List of all benchmarks:
				GoldMajorityVote,
				MajorityVote,
				DawidSkene,
				MMSR,
				Wawa,
				ZeroBasedSkill,
				GLAD

				@click.command()
				@click.option('--dataset-name', default='ionosphere', help='Name of the dataset to be used')
				def main(dataset_name = 'ionosphere'):
					# Loading the dataset
					data, feature_columns = load_data.aim1_3_read_download_UCI_database(WhichDataset=dataset_name)

					# generating the noisy true labels for each crowd worker
					ARLS = {'n_labelers':10,  'low_dis':0.3,   'high_dis':0.9}

					predicted_labels, uncertainty, true_labels, labelers_strength = aim1_3_meauring_probs_uncertainties( data = data, ARLS = ARLS, num_simulations = 20,  feature_columns = feature_columns)

					# Finding the accuracy for all benchmark techniques
					ABTC = Aim1_3_ApplyingBenchmarksToCrowdData(true_labels=true_labels['train'] , n_labelers=ARLS['n_labelers'])
					ABTC.apply_all_benchmarks()

					return ABTC.accuracy, ABTC.f1_score

				accuracy, f1_score = main()
		"""
		self.ground_truth          = ground_truth
		self.crowd_labels          = crowd_labels
		self.crowd_labels_original = crowd_labels.copy()

		for mode in ['train', 'test']:
			self.crowd_labels[mode] = self.reshape_dataframe_into_this_sdk_format(self.crowd_labels[mode])

	def apply(self):
		""" Apply all benchmarks to the input dataset and return the accuracy and f1 score """

		# train    = self.crowd_labels['train']
		# train_gt = self.ground_truth['train']
		test     = self.crowd_labels['test' ]

		# Measuring predicted labels for each benchmar technique:
		test_unique = test.task.unique()

		def exception_handler(func):
			def inner_function(*args, **kwargs):
				try:
					return func(*args, **kwargs)
				except Exception:
					return np.zeros(test_unique.shape)

			return inner_function

		@exception_handler
		def KOS():
			r"""Karger-Oh-Shah aggregation model.

				Iterative algorithm that calculates the log-likelihood of the task being positive while modeling
				the reliabilities of the workers.

				Let $A_{ij}$ be a matrix of answers of worker $j$ on task $i$.
				$A_{ij} = 0$ if worker $j$ didn't answer the task $i$, otherwise $|A_{ij}| = 1$.
				The algorithm operates on real-valued task messages $x_{i \rightarrow j}$  and
				worker messages $y_{j \rightarrow i}$. A task message $x_{i \rightarrow j}$ represents
				the log-likelihood of task $i$ being a positive task, and a worker message $y_{j \rightarrow i}$ represents
				how reliable worker $j$ is.

				On iteration $k$ the values are updated as follows:
				$$
				x_{i \rightarrow j}^{(k)} = \sum_{j^{'} \in \partial i \backslash j} A_{ij^{'}} y_{j^{'} \rightarrow i}^{(k-1)} \\
				y_{j \rightarrow i}^{(k)} = \sum_{i^{'} \in \partial j \backslash i} A_{i^{'}j} x_{i^{'} \rightarrow j}^{(k-1)}
				$$

				Karger, David R., Sewoong Oh, and Devavrat Shah. Budget-optimal task allocation for reliable crowdsourcing systems.
				Operations Research 62.1 (2014): 1-24.

				<https://arxiv.org/abs/1110.3564>

			"""
			return crowdkit_aggregation.KOS().fit_predict(test)

		@exception_handler
		def MACE():
			return crowdkit_aggregation.MACE(n_iter=10).fit_predict_proba(test)[1]

		@exception_handler
		def MajorityVote():
			return crowdkit_aggregation.MajorityVote().fit_predict(test)

		@exception_handler
		def MMSR():
			return crowdkit_aggregation.MMSR().fit_predict(test)

		@exception_handler
		def Wawa():
			return crowdkit_aggregation.Wawa().fit_predict_proba(test)[1]

		@exception_handler
		def ZeroBasedSkill():
			return crowdkit_aggregation.ZeroBasedSkill().fit_predict_proba(test)[1]

		@exception_handler
		def GLAD():
			return crowdkit_aggregation.GLAD().fit_predict_proba(test)[1]

		@exception_handler
		def DawidSkene():
			return crowdkit_aggregation.DawidSkene().fit_predict(test)

		aggregated_labels = pd.DataFrame()
		aggregated_labels['KOS']            = KOS()
		aggregated_labels['MACE']           = MACE()
		aggregated_labels['MMSR']           = MMSR()
		aggregated_labels['Wawa']           = Wawa()
		aggregated_labels['GLAD']           = GLAD()
		aggregated_labels['ZeroBasedSkill'] = ZeroBasedSkill()
		aggregated_labels['MajorityVote']   = MajorityVote()
		aggregated_labels['DawidSkene']     = DawidSkene()

		return aggregated_labels


	@staticmethod
	def reshape_dataframe_into_this_sdk_format(df_predicted_labels):
		"""  Preprocessing the data to adapt to the sdk structure:
		"""

		# Converting labels from binary to integer
		df_crowd_labels: pd.DataFrame = df_predicted_labels.astype(int).copy()

		# Separating the ground truth labels from the crowd labels
		# ground_truth = df_crowd_labels.pop('truth')

		# Stacking all the labelers labels into one column
		df_crowd_labels = df_crowd_labels.stack().reset_index().rename( columns={'level_0': 'task', 'level_1': 'worker', 0: 'label'})

		# Reordering the columns to make it similar to crowd-kit examples
		df_crowd_labels = df_crowd_labels[['worker', 'task', 'label']]

		return df_crowd_labels  # , ground_truth

@dataclass
class Results:
	true_labels      : Dict[str, pd.DataFrame]
	F                : Dict[str, Dict[str, pd.DataFrame]]
	aggregated_labels: pd.DataFrame
	weights_proposed : pd.DataFrame
	weights_Tao      : pd.DataFrame
	labelers_strength: pd.DataFrame
	n_labelers       : int
	metrics 		 : pd.DataFrame

class AIM1_3:
	METHODS_PROPOSED = ['proposed', 'proposed_penalized']
	METHODS_MAIN_BENCHMARKS = ['Tao', 'Sheng']
	METHODS_ALL = ['proposed', 'proposed_penalized'] + ['Tao', 'Sheng'] + Aim1_3_ApplyingBenchmarksToCrowdData.BENCHMARKS

	def __init__(self, config, data, feature_columns, n_labelers=13):

		self.config            = config
		self.data             = data
		self.feature_columns  = feature_columns
		self.n_labelers       = n_labelers
		self.weights_Tao_mean = None
		self.seed             = None
		self.prob_weighted    = None
		self.accuracy         = None
		self.true_labels      = None
		self.uncertainty_all  = None
		self.yhat_benchmark_classifier = None
		self.yhat_proposed_classifier  = None
		self.predicted_labels_all      = None

		self.F                = {}
		self.weights_Tao       = pd.DataFrame()
		self.weights_proposed  = pd.DataFrame()
		self.labelers_strength = pd.DataFrame()

	@staticmethod
	def get_accuracy(aggregated_labels, n_labelers, delta_benchmark, truth):
		""" Measuring accuracy. This result in the same values as if I had measured a weighted majority voting using the "weights" multiplied by "delta" which is the binary predicted labels """

		accuracy = pd.DataFrame(index=[n_labelers])
		for method in AIM1_3.METHODS_ALL:
			accuracy[method] = ((aggregated_labels[method] >= 0.5) == truth).mean(axis=0)

		accuracy['MV_Classifier'] = ( (delta_benchmark.mean(axis=1) >= 0.5) == truth).mean(axis=0)

		return accuracy

	@staticmethod
	def measuring_nu_and_confidence_score(n_labelers, yhat_proposed_classifier, workers_labels, weights_proposed, weights_Tao) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame]:

		def get_pred_and_weights(method_name):
			if method_name in AIM1_3.METHODS_PROPOSED:
				# in the proposed method, unlike Tao and Sheng, we use the output of the trained classifiers as the predicted labels instead of the original worker's labels.
				return yhat_proposed_classifier, weights_proposed[method_name]

			elif method_name == 'Tao':
				return workers_labels, weights_Tao / n_labelers

			elif method_name == 'Sheng':
				return workers_labels, pd.DataFrame(1 / n_labelers, index=weights_Tao.index, columns=weights_Tao.columns)

			raise ValueError(f'Unknown method name: {method_name}')

		def get_F(conf_strategy, delta, weights) -> pd.DataFrame:
			""" Calculating the confidence score for each sample """

			out = pd.DataFrame( index=delta.index )
			# out['truth'] = true_labels['test'].truth

			# Delta is the binarized predicted probabilities
			out['P_pos'] = ( delta * weights).sum(axis=1)
			out['P_neg'] = (~delta * weights).sum(axis=1)

			if conf_strategy == 'freq':
				out['F'] = out[['P_pos','P_neg']].max(axis=1)
				# F[out['P_pos'] < out['P_neg']] = out['P_neg'][out['P_pos'] < out['P_neg']]

			elif conf_strategy == 'beta':
				out['l_alpha'] = 1 + out['P_pos'] * n_labelers
				out['u_beta']  = 1 + out['P_neg'] * n_labelers

				out['k'] = out['l_alpha'] - 1

				# This seems to be equivalent to n_labelers + 1
				out['n'] = ((out['l_alpha'] + out['u_beta']) - 1) #.astype(int)
				# k = l_alpha.floordiv(1)
				# n = (l_alpha+u_beta).floordiv(1) - 1

				get_I = lambda row: bdtrc(row['k'], row['n'], 0.5)
				out['I'] = out.apply(get_I, axis=1)
				# out['I'] = np.nan
				# for index in out['n'].index:
				# 	out['I'][index] = bdtrc(out['k'][index], out['n'][index], 0.5)

				get_F_lambda = lambda row: max(row['I'], 1-row['I'])
				out['F'] = out.apply(get_F_lambda, axis=1)
				# F = I.copy()
				# F[I < 0.5] = (1 - F)[I < 0.5]

			return out

		F = dict(freq={}, beta={})
		aggregated_labels = pd.DataFrame()

		for m in AIM1_3.METHODS_PROPOSED + AIM1_3.METHODS_MAIN_BENCHMARKS:  # Tao: wMV-freq  Sheng: MV-freq

			pred, weights = get_pred_and_weights(m)
			aggregated_labels[m] = (pred * weights).sum(axis=1)

			for strategy in ['freq', 'beta']:
				F[strategy][m] = get_F(strategy, pred, weights)

		return F, aggregated_labels

	@staticmethod
	def core_measurements(data, n_labelers, config, feature_columns) -> Results:
		""" Final pred labels & uncertainties for proposed technique
				dataframe = preds[train, test] * [mv]              <=> {rows: samples,  columns: labelers}
				dataframe = uncertainties[train, test]  {rows: samples,  columns: labelers}

			Final pred labels for proposed benchmarks
				dataframe = preds[train, test] * [simulation_0]    <=> {rows: samples,  columns: labelers} """

		def aim1_3_meauring_probs_uncertainties():
			""" Final pred labels & uncertainties for proposed technique
					dataframe = preds[train, test] * [mv]              <=> {rows: samples,  columns: labelers}
					dataframe = uncertainties[train, test]  {rows: samples,  columns: labelers}

				Final pred labels for proposed benchmarks
					dataframe = preds[train, test] * [simulation_0]    <=> {rows: samples,  columns: labelers} """

			def getting_noisy_manual_labels_for_each_worker(true, labelers_strength=0.5, seed_num=1):

				# setting the random seed
				# np.random.seed(seed_num)

				# number of samples and labelers/workers
				num_samples = true.shape[0]

				# finding a random number for each instance
				true_label_assignment_prob = np.random.random(num_samples)

				# samples that will have an inaccurate true label
				false_samples = true_label_assignment_prob < 1 - labelers_strength

				# measuring the new labels for each labeler/worker
				worker_true = true > 0.5
				worker_true[false_samples] = ~ worker_true[false_samples]

				return worker_true

			def assigning_strengths_randomly_to_each_worker():

				labelers_names = [f'labeler_{j}' for j in range(n_labelers)]

				labelers_strength_array = np.random.uniform(low=config.low_dis, high=config.high_dis, size=n_labelers)

				return pd.DataFrame({'labelers_strength': labelers_strength_array}, index=labelers_names)

			def looping_over_all_labelers(labelers_strength):

				""" Looping over all simulations. this is to measure uncertainties """

				predicted_labels_all_sims = {'train': {}, 'test': {}}
				truth = { 'train': pd.DataFrame(), 'test': pd.DataFrame() }
				uncertainties = { 'train': pd.DataFrame(), 'test': pd.DataFrame() }

				for LB_index, LB in enumerate(labelers_strength.index):

					# Initializationn
					for mode in ['train', 'test']:
						predicted_labels_all_sims[mode][LB] = {}
						truth[mode]['truth'] = data[mode].true.copy()

					# Extracting the simulated noisy manual labels based on the worker's strength
					truth['train'][LB] = getting_noisy_manual_labels_for_each_worker( seed_num=0,  # LB_index,
																					  true=data['train'].true.values,
																					  labelers_strength=labelers_strength.T[LB].values )

					truth['test'][LB] = getting_noisy_manual_labels_for_each_worker( seed_num=1,  # LB_index,
																					 true=data['test'].true.values,
																					 labelers_strength=labelers_strength.T[LB].values )

					SIMULATION_TYPE = 'random_state'

					if SIMULATION_TYPE == 'random_state':
						for sim_num in range(config.num_simulations):
							# training a random forest on the aformentioned labels
							RF = sk_ensemble.RandomForestClassifier(n_estimators=4, max_depth=4, random_state=sim_num)  # n_estimators=4, max_depth=4
							# RF = sklearn.tree.DecisionTreeClassifier(random_state=sim_num)

							RF.fit( X=data['train'][feature_columns], y=truth['train'][LB] )

							# predicting the labels using trained networks for both train and test data
							for mode in ['train', 'test']:
								sim_name = f'simulation_{sim_num}'
								predicted_labels_all_sims[mode][LB][sim_name] = RF.predict(data[mode][feature_columns])

					elif SIMULATION_TYPE == 'multiple_classifier':

						classifiers_list = [
							sklearn.neighbors.KNeighborsClassifier(3),  # type: ignore
							# SVC(kernel="linear", C=0.025),
							sklearn.svm.SVC(gamma=2, C=1),  # type: ignore
							# sklearn.gaussian_process.GaussianProcessClassifier(1.0 * sklearn.gaussian_process.kernels.RBF(1.0)),
							sklearn.tree.DecisionTreeClassifier(max_depth=5),  # type: ignore
							sk_ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
							sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000),  # type: ignore
							sk_ensemble.AdaBoostClassifier(),
							sklearn.naive_bayes.GaussianNB(),  # type: ignore
							sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(),  # type: ignore
						]

						for sim_num, classifier in enumerate(classifiers_list):

							classifier.fit( X=data['train'][feature_columns], y=truth['train'][LB] )

							# predicting the labels using trained networks for both train and test data
							for mode in ['train', 'test']:
								sim_name = f'simulation_{sim_num}'
								predicted_labels_all_sims[mode][LB][sim_name] = classifier.predict(data[mode][feature_columns])

					# Measuring the prediction and uncertainties values after MV over all simulations
					for mode in ['train', 'test']:
						# converting to dataframe
						predicted_labels_all_sims[mode][LB] = pd.DataFrame(predicted_labels_all_sims[mode][LB], index=data[mode].index)

						# predicted probability of each class after MV over all simulations
						predicted_labels_all_sims[mode][LB]['mv'] = ( predicted_labels_all_sims[mode][LB].mean(axis=1) > 0.5)

						# uncertainties for each labeler over all simulations
						uncertainties[mode][LB] = predicted_labels_all_sims[mode][LB].std( axis=1 )

				# reshaping the dataframes
				preds = { 'train': {}, 'test': {} }
				for mode in ['train', 'test']:

					# reversing the order of simulations and labelers. NOTE: for the final experiment I should use simulation_0. if I use the mv, then because the augmented truths keeps changing in each simulation, then with enough simulations, I'll end up witht perfect labelers.
					for i in range(config.num_simulations + 1):

						SM = f'simulation_{i}' if i < config.num_simulations else 'mv'

						preds[mode][SM] = pd.DataFrame()
						for LB in [f'labeler_{j}' for j in range(n_labelers)]:
							preds[mode][SM][LB] = predicted_labels_all_sims[mode][LB][SM]

				return truth, uncertainties, preds

			def adding_accuracy_for_each_labeler(labelers_strength, predicted_labels, true_labels):

				labelers_strength['accuracy-test-classifier'] = 0
				labelers_strength['accuracy-test'] = 0

				for i in range(n_labelers):
					LB = f'labeler_{i}'

					# accuracy of classifier in simulation_0
					labelers_strength.loc[LB, 'accuracy-test-classifier'] = ( predicted_labels['test']['simulation_0'][LB] == true_labels['test'].truth).mean()

					# accuracy of noisy true labels for each labeler
					labelers_strength.loc[LB, 'accuracy-test'] 		     = ( true_labels['test'][LB] == true_labels['test'].truth).mean()

				return labelers_strength

			# setting a random strength for each labeler/worker
			ls = assigning_strengths_randomly_to_each_worker()

			true_labels, uncertainty, predicted_labels = looping_over_all_labelers(labelers_strength=ls)

			labelers_strength = adding_accuracy_for_each_labeler(labelers_strength=ls, predicted_labels=predicted_labels, true_labels=true_labels)

			return predicted_labels, uncertainty, true_labels, labelers_strength

		def aim1_3_measuring_proposed_weights(predicted_labels, predicted_uncertainty):

			# weights       : n_labelers thresh_technique num_methods
			# probs_weighted : num_samples thresh_technique n_labelers

			# To-Do: This is the part where I should measure the prob_mv_binary for different # of workers instead of all of them
			prob_mv_binary = predicted_labels.mean(axis=1) > 0.5

			T1    : dict[str, Any] = {}
			T2    : dict[str, Any] = {}
			w_hat1: dict[str, Any] = {}
			w_hat2: dict[str, Any] = {}

			for workers_name in predicted_labels.columns:
				T1[workers_name] = 1 - predicted_uncertainty[workers_name]

				T2[workers_name] = T1[workers_name].copy()
				T2[workers_name][predicted_labels[workers_name].values != prob_mv_binary.values] = 0

				w_hat1[workers_name] = T1[workers_name].mean(axis=0)
				w_hat2[workers_name] = T2[workers_name].mean(axis=0)

			w_hat = pd.DataFrame([w_hat1, w_hat2], index=AIM1_3.METHODS_PROPOSED).T

			# measuring average weight
			weights = w_hat.divide(w_hat.sum(axis=0), axis=1)

			probs_weighted = pd.DataFrame()
			for method in AIM1_3.METHODS_PROPOSED:
				# probs_weighted[method] =( predicted_uncertainty * weights[method] ).sum(axis=1)
				probs_weighted[method] = (predicted_labels * weights[method]).sum( axis=1 )

			return weights, probs_weighted

		def measuring_Tao_weights_based_on_classifier_labels(delta, noisy_true_labels):
			"""
				tau          : 1 thresh_technique 1
				weights_Tao  : num_samples thresh_technique n_labelers
				W_hat_Tao    : num_samples thresh_technique n_labelers
				z            : num_samples thresh_technique 1
				gamma        : num_samples thresh_technique 1
			"""

			tau = (delta == noisy_true_labels).mean(axis=0)

			# number of labelers
			M = len(delta.columns)

			# number of true and false labels for each class and sample
			true_counts = delta.sum(axis=1)
			false_counts = M - true_counts

			# measuring the "specific quality of instanses"
			s = delta.multiply(true_counts - 1, axis=0) + (~delta).multiply(false_counts - 1, axis=0)
			gamma = (1 + s ** 2) * tau
			W_hat_Tao = gamma.applymap(lambda x: 1 / (1 + np.exp(-x)))
			z = W_hat_Tao.mean(axis=1)

			return W_hat_Tao.divide(z, axis=0)

		def measuring_Tao_weights_based_on_actual_labels(delta, noisy_true_labels):
			"""
				tau          : 1 thresh_technique 1
				weights_Tao  : num_samples thresh_technique n_labelers
				W_hat_Tao    : num_samples thresh_technique n_labelers
				z            : num_samples thresh_technique 1
				gamma        : num_samples thresh_technique 1
			"""

			tau = (delta == noisy_true_labels).mean(axis=0)

			# number of labelers
			M = len(noisy_true_labels.columns)

			# number of true and false labels for each class and sample
			true_counts = noisy_true_labels.sum(axis=1)
			false_counts = M - true_counts

			# measuring the "specific quality of instanses"
			s = noisy_true_labels.multiply(true_counts - 1, axis=0) + (~noisy_true_labels).multiply(false_counts - 1, axis=0)
			gamma = (1 + s ** 2) * tau
			W_hat_Tao = gamma.applymap(lambda x: 1 / (1 + np.exp(-x)))
			z = W_hat_Tao.mean(axis=1)

			return W_hat_Tao.divide(z, axis=0)

		predicted_labels_all, uncertainty_all, true_labels, labelers_strength = aim1_3_meauring_probs_uncertainties()

		yhat_proposed_classifier  = predicted_labels_all['test']['mv'          ]
		yhat_benchmark_classifier = predicted_labels_all['test']['simulation_0']

		# Measuring weights for the proposed technique
		weights_proposed, prob_weighted = aim1_3_measuring_proposed_weights( predicted_labels=yhat_proposed_classifier, predicted_uncertainty=uncertainty_all['test'])

		# Benchmark accuracy measurement
		weights_Tao = measuring_Tao_weights_based_on_actual_labels( delta=yhat_benchmark_classifier, noisy_true_labels=true_labels['test'].drop(columns=['truth']))

		workers_labels = true_labels['test'].drop(columns=['truth'])

		F, aggregated_labels = AIM1_3.measuring_nu_and_confidence_score(n_labelers=n_labelers, workers_labels=workers_labels, yhat_proposed_classifier=yhat_proposed_classifier, weights_proposed=weights_proposed, weights_Tao=weights_Tao)

		# Get the results for other benchmarks
		aggregated_labels_benchmarks = AIM1_3.applying_other_benchmarks(true_labels)
		aggregated_labels = pd.concat( [aggregated_labels, aggregated_labels_benchmarks.copy()], axis=1)

		# Measuring the metrics
		metrics = AIM1_3.get_AUC_ACC_F1(aggregated_labels=aggregated_labels, truth=true_labels['test'].truth)

		# merge labelers_strength and weights
		weights_Tao_mean  = weights_Tao.mean().to_frame().rename(columns={0: 'Tao'})
		labelers_strength = pd.concat( [labelers_strength, weights_proposed * n_labelers, weights_Tao_mean], axis=1)

		return Results(true_labels=true_labels, labelers_strength=labelers_strength, F=F, aggregated_labels=aggregated_labels, weights_proposed=weights_proposed, weights_Tao=weights_Tao, n_labelers=n_labelers, metrics=metrics)

	@staticmethod
	def worker_weight_strength_relation(config, data, feature_columns, seed=0, n_labelers=20) -> pd.DataFrame:

		np.random.seed(seed + 1)
		metric_name = 'weight_strength_relation'
		path_main = config.outputs_path / metric_name / config.dataset_name

		if config.outputs_mode == 'calculate':
			df = AIM1_3.core_measurements(n_labelers=n_labelers, config=config, data=data, feature_columns=feature_columns).labelers_strength.set_index('labelers_strength').sort_index()

			if config.save_outputs:
				LoadSaveFile(path_main / f'{metric_name}.xlsx').dump(df, upload_artifact=config.upload_artifact, artifact_path=metric_name, index=True)

			return df

		elif config.outputs_mode == 'load_local':
			return LoadSaveFile(path_main / f'{metric_name}.xlsx').load(header=0)

		raise ValueError(f'Unknown outputs_mode: {config.outputs_mode}')

	@staticmethod
	def applying_other_benchmarks(true_labels):
		ground_truth = {n: true_labels[n].truth.copy() 				     for n in ['train', 'test']}
		crowd_labels = {n: true_labels[n].drop(columns=['truth']).copy() for n in ['train', 'test']}
		ABTC = Aim1_3_ApplyingBenchmarksToCrowdData(crowd_labels=crowd_labels, ground_truth=ground_truth)
		return ABTC.apply()

	@staticmethod
	def get_AUC_ACC_F1(aggregated_labels, truth): # type: (pd.DataFrame, pd.Series) -> pd.DataFrame

		metrics = pd.DataFrame(index=['AUC', 'ACC', 'F1'], columns=aggregated_labels.columns)

		non_null = ~truth.isnull()
		truth_notnull = truth[non_null].to_numpy()

		if (len(truth_notnull) > 0) and (np.unique(truth_notnull).size == 2):

			for m in aggregated_labels.columns:
				yhat = (aggregated_labels[m] > 0.5).astype(int)[non_null]
				metrics[m]['AUC'] = sk_metrics.roc_auc_score( truth_notnull, yhat)
				metrics[m]['ACC'] = sk_metrics.accuracy_score(truth_notnull, yhat)
				metrics[m]['F1']  = sk_metrics.f1_score( 	  truth_notnull, yhat)

		return metrics

	def get_core_measurements(self, seed=0) -> Results:
		# Setting the random seed
		self.seed = seed
		np.random.seed(seed + 1)
		params = {n: getattr(self, n) for n in ['data', 'n_labelers','config', 'feature_columns']}
		return AIM1_3.core_measurements(**params)


	@staticmethod
	def get_outputs(config, data=None, feature_columns=None, mlflow_setup=None) -> Dict[str, List[Results]]:

		def get_core_results_not_parallel() -> List[Results]:
			# This is not written as one line for loop on purpose (for debugging purposes)
			core_results = []
			for seed in range(config.num_seeds):
				core_results.append(aim1_3.get_core_measurements(seed=seed))

			return core_results


		path = config.outputs_path / 'outputs' / f'{config.dataset_name}.pkl'

		if config.outputs_mode == 'load_local':
			return LoadSaveFile(path).load() # type: ignore

		elif config.outputs_mode == 'calculate':

			outputs = defaultdict(list)
			for nl in tqdm(config.workers_list, desc='looping through different # labelers'):
				aim1_3 = AIM1_3(config=config, n_labelers=nl, data=data, feature_columns=feature_columns)

				# Get the core measurements for all seeds
				if config.parallel_processing:
					with multiprocessing.Pool(processes=config.num_seeds) as pool:
						outputs[f'NL{nl}'] = pool.map( aim1_3.get_core_measurements, list(range(config.num_seeds)))
				else:
					outputs[f'NL{nl}'] = get_core_results_not_parallel()

			# Saving the outputs locally
			if config.save_outputs:
				LoadSaveFile(path).dump(outputs, upload_artifact=config.upload_artifact, artifact_path='outputs/tables/')

			return outputs

		return None  # type: ignore



class AIM1_3_Plot:
	""" Plotting the results"""

	def __init__(self, plot_data: pd.DataFrame):

		self.weight_strength_relation_interpolated = None
		assert type(plot_data) == pd.DataFrame, 'plot_data must be a pandas DataFrame'

		self.plot_data = plot_data

	# def plot(self, plot_data: pd.DataFrame, xlabel='', ylabel='', xticks=True, title='', legend=None, smooth=True, show_markers=True):
	def plot(self, xlabel='', ylabel='', xticks=True, title='', legend=None, smooth=True, interpolation_pt_count=1000, show_markers='proposed'):

		columns = self.plot_data.columns.to_list()
		y       = self.plot_data.values.astype(float)
		x       = self._fixing_x_axis(index=self.plot_data.index)

		xnew, y_smooth = data_interpolation(x=x, y=y, smooth=smooth, interpolation_pt_count=interpolation_pt_count)

		self.weight_strength_relation_interpolated = pd.DataFrame(y_smooth, columns=columns, index=xnew)
		self.weight_strength_relation_interpolated.index.name = 'labelers_strength'

		plt.plot(xnew, y_smooth)
		self._show_markers(show_markers=show_markers, columns=columns, x=x, y=y)

		self._show(x=x, xnew=xnew, y_smooth=y_smooth, xlabel=xlabel, ylabel=ylabel, xticks=xticks, title=title, )
		self._legend(legend=legend, columns=columns)

	@staticmethod
	def _show(x, xnew, y_smooth, xlabel, ylabel, xticks, title):

		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.grid()

		if xticks:
			plt.xticks(xnew)

		plt.show()

		if xticks:
			plt.xticks(x)

		plt.ylim(y_smooth.min() - 0.1, max(1, y_smooth.max()) + 0.1)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.grid(True)

	@staticmethod
	def _legend(legend, columns):

		if legend is None:
			pass
		elif legend == 'empty':
			plt.legend()
		else:
			plt.legend(columns, **legend)

	@staticmethod
	def _fixing_x_axis(index):
		return index.map(lambda x: int(x.replace('NL', ''))) if isinstance(index[0], str) else index.to_numpy()

	@staticmethod
	def _show_markers(show_markers, columns, x, y):
		if show_markers in ('proposed', True):
			cl = [i for i, x in enumerate(columns) if ('proposed' in x) or ('method' in x)]
			plt.plot(x, y[:, cl], 'o')

		elif show_markers == 'all':
			plt.plot(x, y, 'o')


def data_interpolation(x, y, smooth=False, interpolation_pt_count=1000):
	xnew, y_smooth = x, y

	if smooth:
		SMOOTH_METHOD = 'kernel_regression'

		try:

			if SMOOTH_METHOD == 'spline':

				xnew = np.linspace(x.min(), x.max(), interpolation_pt_count)
				spl = make_interp_spline(x, y, k=2)
				y_smooth = spl(xnew)

			elif SMOOTH_METHOD == 'conv':

				filter_size = 5
				filter_array = np.ones(filter_size) / filter_size
				xnew = x.copy()
				y_smooth = np.zeros(list(xnew.shape) + [2])
				for j in range(y.shape[1]):
					y_smooth[:, j] = np.convolve(y[:, j], filter_array, mode='same')

			# elif SMOOTH_METHOD == 'kernel_regression':

			#     xnew = np.linspace(thresh_technique.min(), thresh_technique.max(), interpolation_pt_count)
			#     y_smooth = np.zeros(list(xnew.shape) + [y.shape[1]])
			#     for j in range(y.shape[1]):
			#         kr = statsmodels.nonparametric.kernel_regression.KernelReg(y[:, j], thresh_technique, 'c')
			#         y_smooth[:, j], _ = kr.fit(xnew)

		except Exception as e:
			print(e)
			xnew, y_smooth = x, y

	return xnew, y_smooth


@dataclass
class ClassResultsComparisons:
	outputs                 : dict
	config                   : argparse.Namespace
	findings_confidence_score : dict
	weight_strength_relation : pd.DataFrame


class OutputsForVisualization:

	def __init__(self, config=None, mlflow_setup=None):

		self.config           = config
		self.mlflow_setup     = mlflow_setup
		self.data            = None
		self.feature_columns = None
		self.outputs         = None
		self.accuracy        = None
		self.findings         = None
		self.findings_mean_over_seeds = None
		self.weight_strength_relation = None

	@staticmethod
	def run_for_one_dataset(config, dataset_name='ionosphere', mlflow_setup=None): # type (argparse.Namespace, str, Any) -> ClassResultsComparisons:

		np.random.seed(0)
		config.dataset_name = dataset_name

		# loading the dataset
		data, feature_columns = load_data.aim1_3_read_download_UCI_database(config=config) # type: ignore

		# getting the outputs
		outputs = AIM1_3.get_outputs(config=config, data=data, feature_columns=feature_columns, mlflow_setup=mlflow_setup)

		# measuring the confidence scores
		findings_confidence_score = OutputsForVisualization.get_F_stuff(outputs=outputs, config=config)

		# measuring the worker strength weight relationship for proposed and Tao
		weight_strength_relation = AIM1_3.worker_weight_strength_relation(config=config, data=data, feature_columns=feature_columns, seed=0, n_labelers=20)

		return ClassResultsComparisons(weight_strength_relation=weight_strength_relation, findings_confidence_score=findings_confidence_score, outputs=outputs, config=config)

	@staticmethod
	def run_full_experiment_for_figures(config, mlflow_setup=None): # type: (argparse.Namespace, Any) -> Dict[str, ClassResultsComparisons]
		return {dt: OutputsForVisualization.run_for_one_dataset(dataset_name=dt, config=config, mlflow_setup=mlflow_setup) for dt in config.dataset_list}

	@staticmethod
	def get_F_stuff(outputs, config):

		path_main = config.outputs_path / 'confidence_score' / config.dataset_name

		DICT_KEYS = ['F_all', 'F_pos_all', 'F_mean_over_seeds', 'F_pos_mean_over_seeds']

		def get_Fs_per_nl_per_strategy(strategy, n_workers):

			def get(stuff):
				seeds_list   = list(range(config.num_seeds))
				methods_list = AIM1_3.METHODS_PROPOSED + AIM1_3.METHODS_MAIN_BENCHMARKS
				columns      = pd.MultiIndex.from_product([methods_list, seeds_list], names=['method', 'seed_ix'])
				df			 = pd.DataFrame(columns=columns)

				for (m, sx) in columns:
					df[(m, sx)] = outputs[n_workers][sx].F[strategy][m][stuff].squeeze()

				return df

			inF     = get( 'inF' )
			inF_pos = get( 'P_pos' ) if strategy == 'freq' else get( 'I' )
			inF_mean_over_seeds     = inF.groupby( level=0, axis=1 ).mean()
			inF_pos_mean_over_seeds = inF_pos.groupby( level=0, axis=1 ).mean()

			return inF, inF_pos, inF_mean_over_seeds, inF_pos_mean_over_seeds


		if config.outputs_mode == 'calculate':

			F_dict = { key : dict(freq={}, beta={}) for key in DICT_KEYS }

			for st in ['freq', 'beta']:
				for nl in [f'NL{x}' for x in config.workers_list]:

					F, F_pos, F_mean_over_seeds, F_pos_mean_over_seeds = get_Fs_per_nl_per_strategy( strategy=st, n_workers=nl )

					F_dict['F_all' 			 	  ][st][nl] = F.copy()
					F_dict['F_pos_all' 			  ][st][nl] = F_pos.copy()
					F_dict['F_mean_over_seeds'	  ][st][nl] = F_mean_over_seeds.copy()
					F_dict['F_pos_mean_over_seeds'][st][nl] = F_pos_mean_over_seeds.copy()


			if config.save_outputs:
				for name in DICT_KEYS:
					LoadSaveFile(path_main / f'{name}.pkl').dump(F_dict[name], upload_artifact=config.upload_artifact, artifact_path='')

			return F_dict

		elif config.outputs_mode == 'load_local':
			return { key: LoadSaveFile(path_main / f'{key}.pkl').load()  for key in DICT_KEYS }


		raise ValueError(f'Unknown config.outputs_mode: {config.outputs_mode}')


class Aim1_3_Data_Analysis_Results:

	RENAME_MAPS={'proposed_penalized':'Crowd-Certain' , 'proposed':'Crowd-Certain Without Penalization'}

	def __init__(self, config, mlflow_setup=None):

		self.outputs  = None
		self.accuracy = dict(freq=pd.DataFrame(), beta=pd.DataFrame())

		self.config                 = config
		self.mlflow_setup           = mlflow_setup
		self.config.upload_artifact = (mlflow_setup and config.upload_artifact)

		self.results_all_datasets  = OutputsForVisualization.run_full_experiment_for_figures(config=config, mlflow_setup=mlflow_setup)

	def get_result(self, metric_name='F_all', dataset_name='mushroom', strategy='freq' , nl='NL3', seed_ix=0, method_name='proposed', data_mode='test'):

		def drop_proposed_rename_crowd_certain(df, orient='columns'):

			if orient == 'columns':
				return df.drop(columns=['proposed']).rename(columns={n:self.RENAME_MAPS[n] for n in ['proposed_penalized']})
			else:
				return df.drop(index=['proposed']).rename(index={n:self.RENAME_MAPS[n] for n in ['proposed_penalized']})

		def get_metrics_mean_over_seeds(dataset_name1, n_workers) -> pd.DataFrame:
			seed_list = list(range(self.config.num_seeds))
			df_all    = pd.DataFrame(columns=pd.MultiIndex.from_product([seed_list, ['AUC', 'ACC', 'F1']], names=['s', 'metric']))

			for s in range( self.config.num_seeds ):
				df_all[s] = self.results_all_datasets[dataset_name1].outputs[n_workers][s].metrics.T.astype( float )

			return df_all.groupby(level='metric', axis=1).mean()

		if metric_name in ['F_all', 'F_pos_all', 'F_mean_over_seeds', 'F_pos_mean_over_seeds']:

			df = self.results_all_datasets[dataset_name].findings_confidence_score[metric_name][strategy][nl]

			if strategy == 'freq':
				df = 1 - df

			df['truth'] = self.results_all_datasets[dataset_name].outputs[nl][seed_ix].true_labels[data_mode].truth
			return df

		elif metric_name in ['F_eval_one_dataset_all_labelers', 'F_eval_one_worker_all_datasets']:
			return self.get_evaluation_metrics_for_confidence_scores(metric_name=metric_name, dataset_name=dataset_name, nl=nl)

		elif metric_name == 'weight_strength_relation': # 'df_weight_stuff'
			value_vars = list(self.RENAME_MAPS.values()) + ['Tao']

			wwr = pd.DataFrame()
			for dt in self.config.dataset_list:

				df = (self.results_all_datasets[dt].weight_strength_relation
						.rename(columns=self.RENAME_MAPS)
						.melt(id_vars=['labelers_strength'], value_vars=value_vars, var_name='Method', value_name='Weight'))

				wwr = pd.concat([wwr, df.assign(dataset_name=dt)], axis=0)

			return wwr


		elif metric_name in ['F', 'aggregated_labels','true_labels']:

			if metric_name == 'F':
				assert method_name in ['proposed', 'proposed_penalized', 'Tao', 'Sheng']
				return self.results_all_datasets[dataset_name].outputs[nl][seed_ix].F[strategy][method_name]

			elif metric_name in ['aggregated_labels']:
				return self.results_all_datasets[dataset_name].outputs[nl][seed_ix].aggregated_labels

			elif metric_name == 'true_labels':
				assert data_mode in ['train', 'test']
				return self.results_all_datasets[dataset_name].outputs[nl][seed_ix].true_labels[data_mode]

		elif 'metric' in metric_name:

			if metric_name == 'metrics_per_dataset_per_worker_per_seed':
				return self.results_all_datasets[dataset_name].outputs[nl][seed_ix].metrics.T.astype(float)

			elif metric_name == 'metrics_mean_over_seeds_per_dataset_per_worker':
				df = get_metrics_mean_over_seeds(dataset_name, nl)
				return drop_proposed_rename_crowd_certain(df, orient='index')

			elif metric_name == 'metrics_all_datasets_workers':
				workers_list = [f'NL{i}' for i in self.config.workers_list]

				columns = pd.MultiIndex.from_product([['ACC','AUC','F1'], self.config.dataset_list, workers_list], names=['metric', 'dataset', 'workers'])
				df = pd.DataFrame(columns=columns)

				for dt in self.config.dataset_list:
					for nl in workers_list:
						df_temp = get_metrics_mean_over_seeds(dt, nl)
						df_temp = drop_proposed_rename_crowd_certain(df_temp, orient='index')

						for metric in ['ACC','AUC','F1']:
							df[(metric, dt, nl)] = df_temp[metric].copy()

				return df

	def get_evaluation_metrics_for_confidence_scores(self, metric_name='F_eval_one_dataset_all_labelers', dataset_name='ionosphere', nl='NL3'):

		from sklearn.metrics import brier_score_loss

		def ece_score(y_true, conf_scores, n_bins=10):
			"""Compute ECE"""
			bin_boundaries = np.linspace(0, 1, n_bins + 1)
			bin_lowers = bin_boundaries[:-1]
			bin_uppers = bin_boundaries[1:]

			accuracies = []
			confidences = []
			for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
				# Filter out data points from the bin
				in_bin = np.logical_and(bin_lower <= conf_scores,
										conf_scores < bin_upper)
				prop_in_bin = in_bin.mean()

				if prop_in_bin > 0:
					accuracy_in_bin = y_true[in_bin].mean()
					confidence_in_bin = conf_scores[in_bin].mean()

					accuracies.append(accuracy_in_bin)
					confidences.append(confidence_in_bin)

			accuracies = np.array(accuracies)
			confidences = np.array(confidences)

			ece = np.sum(np.abs(accuracies - confidences) * confidences)
			return ece

		if metric_name == 'F_eval_one_dataset_all_labelers':
			target_list = [f'NL{x}' for x in self.config.workers_list]
			target_name = 'n_labelers'
			get_params = lambda i: dict(nl=i, dataset_name=dataset_name)
		elif metric_name == 'F_eval_one_worker_all_datasets':
			target_list = self.config.dataset_list
			target_name = 'dataset_name'
			get_params = lambda i: dict(nl=nl, dataset_name=i)
		else:
			raise ValueError('metric_name should be either workers or datasets')

		proposed_technique_name = 'Crowd-Certain'
		index   = pd.MultiIndex.from_product([['freq' , 'beta'], [proposed_technique_name, 'Tao', 'Sheng']], names=['strategy', 'technique'])
		columns = pd.MultiIndex.from_product([['ece score', 'brier score loss'], target_list], names=['metric', target_name])
		df_cs   = pd.DataFrame(columns=columns, index=index)

		for st in ['freq' , 'beta']:
			for ix in target_list:
				conf_scores = self.get_result(metric_name='F_pos_mean_over_seeds', strategy=st, **get_params(ix))
				conf_scores = conf_scores[['proposed_penalized', 'Tao', 'Sheng', 'truth']].rename(columns={'proposed_penalized': proposed_technique_name})

				for m in [proposed_technique_name, 'Tao', 'Sheng']:
					df_cs[('ece score'       ,ix)][(st, m)] = ece_score(conf_scores.truth, conf_scores[m])
					df_cs[('brier score loss',ix)][(st, m)] = brier_score_loss(conf_scores.truth, conf_scores[m])

		return df_cs.astype(float)

	def save_outputs(self, filename, relative_path, dataframe=None):

		# output path
		path = self.config.outputs_path / relative_path / filename
		path.mkdir(parents=True, exist_ok=True)

		# Save the plot
		for suffix in ['png', 'eps', 'svg', 'pdf']:
			plt.savefig(path / f'{filename}.{suffix}', format=suffix, dpi=300, bbox_inches='tight')

		# Save the sheet
		if dataframe is not None: LoadSaveFile( path / f'{filename}.xlsx' ).dump( dataframe, index=True )


	def figure_weight_quality_relation(self, aspect=1.5, font_scale=1.8, fontsize=15, relative_path='final_figures', height=4):

		metric_name = 'weight_strength_relation'
		df: pd.DataFrame = self.get_result(metric_name=metric_name)  # type: ignore

		sns.set( palette='colorblind', style='darkgrid', context='paper', font_scale=font_scale)

		p = sns.lmplot(data=df, legend=True, hue='Method', order=3, legend_out=False, x="labelers_strength", y="Weight", col='dataset_name', col_wrap=3, height=height, aspect=aspect, sharex=True, sharey=True, ci=None)

		p.set_xlabels(r"Probability Threshold ($\pi_\alpha^{(k)}$)" , fontsize=fontsize)
		p.set_ylabels(r"Estimated Weight ($\omega_\alpha^{(k)}$)"   , fontsize=fontsize)
		p.set_titles(col_template="{col_name}", fontweight='bold'   , fontsize=fontsize)
		p.fig.suptitle('Estimated Weight vs. Probability Threshold'  ,  fontsize=int(1.5*fontsize), fontweight='bold')
		p.tight_layout()
		sns.move_legend(p,"lower right", bbox_to_anchor=(0.75, 0.1) , bbox_transform=p.fig.transFigure)

		# Saving the Figure & Sheet
		self.save_outputs( filename=f'figure_{metric_name}', relative_path=relative_path, dataframe=df )

	def figure_metrics_mean_over_seeds_per_dataset_per_worker(self, metric='ACC', nl=3, figsize=(10, 10), font_scale=1.8, fontsize=20, relative_path='final_figures'):

		metric_name='metrics_mean_over_seeds_per_dataset_per_worker'

		df = pd.DataFrame()
		for dt in self.config.dataset_list:
			df[dt] = self.get_result( metric_name=metric_name, dataset_name=dt, nl=f'NL{nl}')[metric] # type: ignore


		fig = plt.figure(figsize=figsize)
		sns.set(font_scale=font_scale, palette='colorblind', style='darkgrid', context='paper')
		sns.heatmap(df.T, annot=True, fmt='.2f', cmap='Blues',  cbar=True, robust=True)

		fig.suptitle(f'{metric} for NL{nl} ({nl} Workers)', fontsize=int(1.5*fontsize), fontweight='bold')
		plt.tight_layout()

		# Saving the Figure & Sheet
		self.save_outputs( filename=f'figure_{metric_name}_{metric}', relative_path=relative_path, dataframe=df )

	def figure_metrics_all_datasets_workers(self, workers_list=['NL3', 'NL4', 'NL5'], figsize=(15, 15), font_scale=1.8, fontsize=20, relative_path='final_figures'):

		sns.set(font_scale=font_scale, palette='colorblind', style='darkgrid', context='paper')

		metric_name  = 'metrics_all_datasets_workers'
		metrics_list = ['ACC', 'F1', 'AUC']

		df: pd.DataFrame = self.get_result(metric_name=metric_name) # type: ignore

		fig, axes = plt.subplots(nrows=len(workers_list), ncols=len(metrics_list), figsize=figsize, sharex=True, sharey=True, squeeze=True)
		for i2, metric in enumerate(metrics_list):
			df_per_nl = df[metric].groupby(level=1, axis=1)
			for i1, nl in enumerate(workers_list):
				sns.boxplot(data=df_per_nl.get_group( nl).T, orient='h', ax=axes[i1, i2])


		for i2, metric in enumerate(metrics_list):
			axes[2, i2].set_xlabel(metric, fontsize=fontsize, fontweight='bold', labelpad=20)

		for i1, nl in enumerate(workers_list):
			axes[i1, 0].set_ylabel(nl, fontsize=fontsize, fontweight='bold', labelpad=20)

		fig.suptitle('Metrics Over All Datasets', fontsize=int(1.5*fontsize), fontweight='bold')
		plt.tight_layout()

		# Saving the Figure & Sheet
		self.save_outputs( filename=f'figure_{metric_name}', relative_path=relative_path, dataframe=df )

	def figure_F_heatmap(self, metric_name='F_eval_one_dataset_all_labelers', dataset_name='ionosphere', nl='NL3', fontsize=20, font_scale=1.8, figsize=(13, 11), relative_path='final_figures'):

		sns.set(font_scale=font_scale, palette='colorblind', style='darkgrid', context='paper')

		# Set labels for the columns
		if metric_name == 'F_eval_one_dataset_all_labelers':
			filename  = f'heatmap_F_evals_{dataset_name}_all_labelers'
			suptitle = f'Confidence Score Evaluation for {dataset_name}'

		elif metric_name == 'F_eval_one_worker_all_datasets':
			filename  = f'heatmap_F_evals_all_datasets_{nl}'
			suptitle = f'Confidence Score Evaluation for {nl.split("NL")[1]} Workers'

		else:
			raise ValueError('metric_name does not exist')


		df = self.get_result(metric_name=metric_name, dataset_name=dataset_name, nl=nl).round(3)


		fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex=True, sharey=True, squeeze=True) # type: ignore

		# Create the heatmaps
		sns.heatmap(df['ece score'].T['freq']		, ax=axes[0, 0], annot=True, fmt='.2f', cmap='Reds',  cbar=False, robust=True)
		sns.heatmap(df['ece score'].T['beta']		, ax=axes[0, 1], annot=True, fmt='.2f', cmap='Reds',  cbar=True,  robust=True)
		sns.heatmap(df['brier score loss'].T['freq'], ax=axes[1, 0], annot=True, fmt='.2f', cmap='Blues', cbar=False, robust=True)
		sns.heatmap(df['brier score loss'].T['beta'], ax=axes[1, 1], annot=True, fmt='.2f', cmap='Blues', cbar=True,  robust=True)

		# Add a title to each subplot
		axes[0, 0].set_title("Freq", fontsize=fontsize, fontweight='bold')
		axes[0, 1].set_title("Beta", fontsize=fontsize, fontweight='bold')

		# Set labels for the rows
		axes[0, 0].set_ylabel("ECE"       , fontsize=fontsize, fontweight='bold', labelpad=20)
		axes[1, 0].set_ylabel("Brier Score", fontsize=fontsize, fontweight='bold', labelpad=20)
		axes[0, 1].set_ylabel('')
		axes[1, 1].set_ylabel('')

		axes[1, 0].set_xlabel('')
		axes[1, 1].set_xlabel('')
		axes[0, 0].set_xlabel('')
		axes[0, 1].set_xlabel('')


		fig.suptitle(suptitle, fontsize=int(1.5*fontsize), fontweight='bold')
		plt.tight_layout()

		self.save_outputs( filename=f'figure_{filename}', relative_path=relative_path, dataframe=df.T )



# def get_p_value_kappa_cohen_d_BF10(df, node): # type: (pd.DataFrame, str) -> None
#
#     # Perform the independent samples t-test
#     df.loc['t_stat',node], df.loc['p_value',node] = stats.ttest_ind( baseline.yhat[node], proposed.yhat[node])
#
#     # kappa inter rater metric
#     df.loc['kappa',node] = cohen_kappa_score(baseline.yhat[node], proposed.yhat[node])
#
#     df_ttest = pg.ttest(baseline.yhat[node], proposed.yhat[node])
#     df.loc['power',node]   = df_ttest['power'].values[0]
#     df.loc['cohen-d',node] = df_ttest['cohen-d'].values[0]
#     df.loc['BF10',node]    = df_ttest['BF10'].values[0]