import pandas as pd
import numpy as np
import pathlib
import os
from typing import Dict, List, Tuple, Union
from sklearn import preprocessing
from crowd_certain.utilities.config.params import DatasetNames
from ucimlrepo import fetch_ucirepo, list_available_datasets

from crowd_certain.utilities.config.settings import Settings

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def find_dataset_path(base_path: pathlib.Path = None, debug: bool = False) -> pathlib.Path:
    """
    Find the correct dataset path, trying alternative paths if the specified one doesn't exist.

    Args:
        base_path: The base path to check first
        verbose: Whether to print status messages

    Returns:
        The correct dataset path
    """
    # If base_path exists, return it immediately
    if base_path is not None and base_path.exists():
        if debug:
            print(f"Using existing dataset path: {base_path}")
        return base_path

    # If base_path doesn't exist, check alternative paths
    if debug:
        print(f"Dataset directory {base_path} does not exist. Checking alternative paths...")

    # Try alternative paths if the specified one doesn't exist
    alt_paths = [

        # Current directory and subdirectories
        pathlib.Path('datasets'),
        pathlib.Path('crowd_certain/datasets'),

        # Relative to the current file
        pathlib.Path(__file__).parent.parent / 'datasets',

        # Relative to current working directory
        pathlib.Path.cwd() / 'datasets',
        pathlib.Path.cwd() / 'crowd_certain' / 'datasets',

        # Absolute paths that might exist in the project
        pathlib.Path.home() / 'crowd_certain' / 'datasets'
    ]

    # Check each alternative path
    for alt_path in alt_paths:
        if alt_path.exists():
            if debug:
                print(f"Using alternative dataset path: {alt_path}")
            return alt_path

    # If no alternative paths exist, create and return the base_path
    if debug:
        print(f"No alternative dataset paths found.")

    if base_path is None:
        base_path = pathlib.Path.cwd() / 'datasets'

    try:
        os.makedirs(base_path, exist_ok=True)
        if debug:
            print(f"Successfully created dataset directory: {base_path}")

    except Exception as e:
        print(f"Error creating dataset directory: {str(e)}")

    return base_path

def load_dataset(config: Settings, dataset_name: DatasetNames = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load a dataset from the UCI Machine Learning Repository using the ucimlrepo library.
    First checks if the dataset is available locally, and only downloads it if not found.

    Args:
        config: Configuration object with dataset settings
        dataset_name: Optional dataset name to override the one in config

    Returns:
        Tuple containing:
            - Dictionary with 'train' and 'test' DataFrames
            - List of feature column names
    """
    dataset_name = dataset_name or config.dataset.dataset_name

    # Get the UCI dataset ID directly from the DatasetNames enum
    dataset_id = dataset_name.uci_id

    print(f"Loading dataset: {dataset_name.value} (ID: {dataset_id}) from UCI ML Repository")

    # First try to load from local cache
    try:
        return load_from_local_cache(config, dataset_name)
    except Exception as local_e:
        print(f"Local dataset not found or could not be loaded: {str(local_e)}")
        print("Downloading from UCI ML Repository...")

    # If local cache failed, download from UCI
    try:
        # Fetch the dataset from ucimlrepo
        dataset = fetch_ucirepo(id=dataset_id)

        # Get features and targets
        X = dataset.data.features
        y = dataset.data.targets

        # Combine features and targets
        if isinstance(y, pd.DataFrame) and len(y.columns) == 1:
            y_col_name = y.columns[0]
            data_raw = pd.concat([X, y.rename(columns={y_col_name: 'true'})], axis=1)
        elif isinstance(y, pd.Series):
            data_raw = pd.concat([X, pd.DataFrame({'true': y})], axis=1)
        else:
            # For multiple target columns, use the first one
            raise ValueError(f"Dataset {dataset_name.value} has multiple target columns. Not supported.")

        # Process the dataset based on its type
        data_raw, feature_columns = process_dataset(data_raw=data_raw, dataset_name=dataset_name)

        # Save the processed dataset to local cache for future use
        save_to_local_cache(config, dataset_name, data_raw)

        # Split into train and test sets
        data = separate_train_test(data_raw, train_frac=config.dataset.train_test_ratio,
                                  random_state=config.dataset.random_state)

        return data, feature_columns

    except Exception as e:
        print(f"Error downloading dataset from ucimlrepo: {str(e)}")
        raise ValueError(f"Could not load dataset {dataset_name.value} from ucimlrepo or local cache")

def process_dataset(data_raw: pd.DataFrame, dataset_name: DatasetNames) -> Tuple[pd.DataFrame, List[str]]:
    """
    Process the dataset based on its type.

    Args:
        data_raw: Raw DataFrame with features and target
        dataset_name: Name of the dataset

    Returns:
        Tuple containing:
            - Processed DataFrame
            - List of feature column names
    """
    # Get feature columns (all columns except 'true')
    feature_columns = [col for col in data_raw.columns if col != 'true']

    # Convert categorical variables to numeric if needed
    # Process categorical columns
    for col in data_raw.columns:
        if data_raw[col].dtype == 'object':
            # Define target column mappings for known datasets
            target_mappings = {
                DatasetNames.CHESS        : {'won': 1, 'nowin': 0},
                DatasetNames.MUSHROOM     : {'e': 1, 'p': 0},
                DatasetNames.HEART        : {1: 1, 0: 0},
                DatasetNames.TIC_TAC_TOE  : {'positive': 1, 'negative': 0},
                DatasetNames.VOTE         : {'democrat': 1, 'republican': 0},
                DatasetNames.IONOSPHERE   : {'g': 1, 'b': 0},
                DatasetNames.BREAST_CANCER: {'M': 1, 'B': 0},
                DatasetNames.SONAR        : {'M': 1, 'R': 0}
            }

            if col == 'true':
                # Apply predefined mapping if available, otherwise use label encoding
                if dataset_name in target_mappings:
                    data_raw[col] = data_raw[col].map(target_mappings[dataset_name])
                else:
                    data_raw[col] = preprocessing.LabelEncoder().fit_transform(data_raw[col])
            else:
                # Use label encoding for feature columns
                data_raw[col] = preprocessing.LabelEncoder().fit_transform(data_raw[col])

    # Dataset-specific processing
    if dataset_name == DatasetNames.IRIS:
        # Keep only two classes for binary classification
        data_raw = data_raw[data_raw['true'] != 2]

    elif dataset_name == DatasetNames.WAVEFORM:
        # Keep only classes 1 and 2, remap to 0 and 1
        data_raw = data_raw[data_raw['true'] != 0]
        data_raw['true'] = data_raw['true'].replace({1: 0, 2: 1})

    elif dataset_name == DatasetNames.CAR:
        # Remap classes 2 and 3 to 1 (acceptable, good, very good -> 1)
        data_raw['true'] = data_raw['true'].replace({2: 1, 3: 1})

    # Handle missing values
    data_raw.replace(2147483648, np.nan, inplace=True)
    data_raw.replace(-2147483648, np.nan, inplace=True)

    # Remove columns with only one unique value
    cols_to_drop = []
    for col in feature_columns:
        if col in data_raw.columns and len(data_raw[col].unique()) == 1:
            cols_to_drop.append(col)

    if cols_to_drop:
        data_raw.drop(columns=cols_to_drop, inplace=True)
        feature_columns = [col for col in feature_columns if col not in cols_to_drop]

    return data_raw, feature_columns

def separate_train_test(data_raw: pd.DataFrame, train_frac: float = 0.8, random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Split the dataset into training and testing sets.

    Args:
        data_raw: DataFrame with features and target
        train_frac: Fraction of data to use for training
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with 'train' and 'test' DataFrames
    """
    train = data_raw.sample(frac=train_frac, random_state=random_state).sort_index()
    test = data_raw.drop(train.index)
    return {'train': train, 'test': test}

def load_from_local_cache(config: Settings, dataset_name: DatasetNames) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load a dataset from the local cache.

    Args:
        config: Configuration object with dataset settings
        dataset_name: Name of the dataset

    Returns:
        Tuple containing:
            - Dictionary with 'train' and 'test' DataFrames
            - List of feature column names
    """
    # Get or create the dataset directory
    base_dataset_dir = find_dataset_path(config.dataset.path_all_datasets, debug=False)

    # Look for dataset in possible locations - only using the specified formats
    dataset_name_str = dataset_name.value
    possible_paths = [
        base_dataset_dir / f'{dataset_name_str}/{dataset_name_str}.data',
        base_dataset_dir / f'{dataset_name_str}/{dataset_name_str}.csv',
    ]

    # Print all paths being checked (for debugging)
    print(f"Checking the following paths for dataset {dataset_name_str}:")
    for path in possible_paths:
        if path.exists():
            print(f"  - {path} (exists: {path.exists()})")

    # Find the first existing dataset file
    dataset_file = next((path for path in possible_paths if path.exists()), None)

    if not dataset_file:
        raise FileNotFoundError(f"Could not find dataset {dataset_name_str} in local cache. Checked paths: {[str(p) for p in possible_paths]}")

    print(f"Found dataset in local cache at {dataset_file}")

    # Read the data based on file extension
    try:
        if dataset_file.suffix == '.csv':
            data_raw = pd.read_csv(dataset_file)
        elif dataset_file.suffix == '.data':
            data_raw = pd.read_csv(dataset_file, header=None)
            # Ensure the target column is named 'true'
            if 'true' not in data_raw.columns:
                data_raw = data_raw.rename(columns={data_raw.columns[-1]: 'true'})
        else:
            raise ValueError(f"Unsupported file format: {dataset_file.suffix}")
    except Exception as e:
        raise ValueError(f"Error reading dataset file {dataset_file}: {str(e)}")

    # Process the dataset
    data_raw, feature_columns = process_dataset(data_raw=data_raw, dataset_name=dataset_name)

    # Split into train and test sets
    data = separate_train_test(data_raw=data_raw, train_frac=config.dataset.train_test_ratio, random_state=config.dataset.random_state)

    return data, feature_columns

def save_to_local_cache(config: Settings, dataset_name: DatasetNames, data_raw: pd.DataFrame) -> None:
    """
    Save a dataset to the local cache for future use.

    Args:
        config: Configuration object with dataset settings
        dataset_name: Name of the dataset
        data_raw: DataFrame with the processed dataset
    """
    # Define the base dataset directory
    base_dataset_dir = find_dataset_path(config.dataset.path_all_datasets, debug=False)

    # Create a standardized path for the dataset - always using the specified format
    dataset_name_str = dataset_name.value
    dataset_dir = base_dataset_dir / dataset_name_str
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = dataset_dir / f"{dataset_name_str}.csv"

    # Save the dataset
    try:
        data_raw.to_csv(dataset_file, index=False)
        print(f"Saved dataset to local cache at {dataset_file}")
    except Exception as e:
        print(f"Warning: Could not save dataset to local cache: {str(e)}")

        # Try an alternative location if the primary one fails, still using the specified format
        try:
            alt_base_dir = find_dataset_path(pathlib.Path(__file__).parent.parent / 'datasets', debug=False)
            alt_dir = alt_base_dir / dataset_name_str
            alt_dir.mkdir(parents=True, exist_ok=True)
            alt_file = alt_dir / f"{dataset_name_str}.csv"
            data_raw.to_csv(alt_file, index=False)
            print(f"Saved dataset to alternative location: {alt_file}")
        except Exception as alt_e:
            print(f"Failed to save dataset to alternative location: {str(alt_e)}")



