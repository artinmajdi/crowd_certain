"""
HDF5 Storage module for handling complex data structures in the crowd_certain package.

This module provides a robust way to store and retrieve complex nested data structures
including pandas DataFrames, numpy arrays, dictionaries, and custom objects using HDF5 format,
which offers better performance, compression, and organization compared to pickle files.
"""

import pickle
import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

class HDF5Storage:
    """
    A modern storage class for handling complex nested data using HDF5.

    This class provides methods to save and load various data types to/from HDF5 files,
    including pandas DataFrames, numpy arrays, dictionaries, and custom dataclasses.
    It handles complex nested structures by using a hierarchical organization within
    the HDF5 file.

    Parameters
    ----------
    path : pathlib.Path or str
        The file path where the HDF5 file should be located.

    Methods
    -------
    save(data, group_path='/', metadata=None)
        Save data to the specified group path in the HDF5 file.

    load(group_path='/', load_metadata=False)
        Load data from the specified group path in the HDF5 file.

    save_dataframe(df, group_path)
        Save a pandas DataFrame to the specified group path.

    load_dataframe(group_path)
        Load a pandas DataFrame from the specified group path.

    list_groups()
        List all available groups in the HDF5 file.
    """
    def __init__(self, path):
        self.path = Path(path)
        if not str(self.path).endswith('.h5'):
            self.path = self.path.with_suffix('.h5')

    def _ensure_parent_exists(self):
        """Ensure that the parent directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data, group_path='/', metadata=None):
        """
        Save data to an HDF5 file.

        This method handles various data types and saves them appropriately:
        - pandas DataFrames are saved using the pandas HDF5 interface
        - numpy arrays are saved directly
        - dictionaries are saved as groups with each key as a subgroup/dataset
        - lists are saved as arrays or groups with numbered items
        - primitive types are saved as attributes or datasets
        - custom objects are serialized using pickle

        Parameters
        ----------
        data : any
            The data to save
        group_path : str, default '/'
            The path within the HDF5 file to save the data
        metadata : dict, optional
            Additional metadata to store with the data
        """
        self._ensure_parent_exists()

        # Handle None value
        if data is None:
            with h5py.File(self.path, 'a') as hf:
                if group_path in hf:
                    del hf[group_path]
                g = hf.create_group(group_path)
                g.attrs['type'] = 'none'
            return

        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            return self.save_dataframe(data, group_path, metadata)

        # Handle pandas Series
        elif isinstance(data, pd.Series):
            return self.save_series(data, group_path, metadata)

        # Handle numpy array
        elif isinstance(data, np.ndarray):
            with h5py.File(self.path, 'a') as hf:
                if group_path in hf:
                    del hf[group_path]
                hf.create_dataset(group_path, data=data, compression="gzip", compression_opts=9)
                if metadata:
                    for key, value in metadata.items():
                        hf[group_path].attrs[key] = value

        # Handle dictionaries
        elif isinstance(data, dict):
            with h5py.File(self.path, 'a') as hf:
                # Create or clear the group
                if group_path in hf:
                    del hf[group_path]
                g = hf.create_group(group_path)
                g.attrs['type'] = 'dict'

                # Save keys as a dataset for easier retrieval
                keys_list = list(data.keys())
                try:
                    # Try to save keys directly if they're simple types
                    g.create_dataset('__keys__', data=keys_list)
                except Exception:
                    # Otherwise, serialize them
                    keys_serialized = json.dumps([str(k) for k in keys_list])
                    g.create_dataset('__keys__', data=np.array([keys_serialized.encode('utf-8')]))
                    g.attrs['keys_serialized'] = True

                # Save each item
                for key, value in data.items():
                    key_str = str(key)
                    safe_key = key_str.replace('/', '_')  # Make key safe for HDF5 paths
                    self.save(value, f"{group_path}/{safe_key}")
                    g[safe_key].attrs['original_key'] = key_str

                # Add metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        g.attrs[key] = value

        # Handle lists and tuples
        elif isinstance(data, (list, tuple)):
            with h5py.File(self.path, 'a') as hf:
                if group_path in hf:
                    del hf[group_path]
                g = hf.create_group(group_path)
                g.attrs['type'] = 'list' if isinstance(data, list) else 'tuple'
                g.attrs['length'] = len(data)

                try:
                    # Try to save as a single dataset if all elements are compatible
                    if len(data) > 0 and all(isinstance(item, (int, float, bool, str)) for item in data):
                        if all(isinstance(item, str) for item in data):
                            dt = h5py.special_dtype(vlen=str)
                            g.create_dataset('items', data=np.array(data, dtype=dt))
                        else:
                            g.create_dataset('items', data=np.array(data))
                    else:
                        # Save as individual items
                        for i, item in enumerate(data):
                            self.save(item, f"{group_path}/item_{i}")
                except Exception as e:
                    # Fall back to saving as individual items
                    for i, item in enumerate(data):
                        self.save(item, f"{group_path}/item_{i}")

                # Add metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        g.attrs[key] = value

        # Handle primitive types
        elif isinstance(data, (int, float, bool, str)):
            with h5py.File(self.path, 'a') as hf:
                if group_path in hf:
                    del hf[group_path]

                dtype = None
                if isinstance(data, str):
                    dtype = h5py.special_dtype(vlen=str)

                dataset = hf.create_dataset(group_path, data=np.array([data], dtype=dtype))
                dataset.attrs['type'] = type(data).__name__

                # Add metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        dataset.attrs[key] = value

        # Handle other classes (custom objects)
        else:
            with h5py.File(self.path, 'a') as hf:
                if group_path in hf:
                    del hf[group_path]

                # For custom classes, use pickle serialization
                pickled_data = pickle.dumps(data)
                dset = hf.create_dataset(group_path, data=np.void(pickled_data))
                dset.attrs['type'] = 'pickled'
                dset.attrs['class'] = type(data).__name__

                # Add metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        dset.attrs[key] = value

    def load(self, group_path='/', load_metadata=False):
        """
        Load data from an HDF5 file.

        This method retrieves data from the specified group path, handling:
        - pandas DataFrames
        - numpy arrays
        - dictionaries
        - lists/tuples
        - primitive types
        - custom pickled objects

        Parameters
        ----------
        group_path : str, default '/'
            The path within the HDF5 file to load data from
        load_metadata : bool, default False
            Whether to return metadata alongside the data

        Returns
        -------
        data : Any
            The loaded data
        metadata : dict, optional
            Metadata associated with the data, returned if load_metadata=True
        """
        if not self.path.exists():
            if load_metadata:
                return None, {}
            return None

        with h5py.File(self.path, 'r') as hf:
            if group_path not in hf:
                if load_metadata:
                    return None, {}
                return None

            # Get item from file
            item = hf[group_path]

            # Load metadata if requested
            metadata = {}
            if load_metadata:
                for key, value in item.attrs.items():
                    metadata[key] = value

            # Check if item is a group or dataset
            if isinstance(item, h5py.Group):
                # Check if this is a DataFrame
                if 'df_index' in item and 'df_columns' in item:
                    result = self.load_dataframe(group_path)
                    if load_metadata:
                        return result, metadata
                    return result

                # Check if this is a Series
                elif 'series_index' in item:
                    result = self.load_series(group_path)
                    if load_metadata:
                        return result, metadata
                    return result

                # Check if this is a dictionary
                elif item.attrs.get('type') == 'dict':
                    keys_dataset = item['__keys__']

                    if item.attrs.get('keys_serialized', False):
                        # Deserialize keys
                        keys_json = keys_dataset[0].decode('utf-8')
                        keys = json.loads(keys_json)
                    else:
                        # Get keys directly
                        keys = list(keys_dataset[()])

                    result = {}
                    for child_name in item:
                        if child_name == '__keys__':
                            continue

                        child = item[child_name]
                        original_key = child.attrs.get('original_key', child_name)

                        # Find the correct key type from the keys list
                        key = None
                        for k in keys:
                            if str(k) == original_key:
                                key = k
                                break

                        if key is None:
                            key = original_key

                        result[key] = self.load(f"{group_path}/{child_name}")

                    if load_metadata:
                        return result, metadata
                    return result

                # Check if this is a list or tuple
                elif item.attrs.get('type') in ('list', 'tuple'):
                    if 'items' in item:
                        # List was stored as a single dataset
                        result = list(item['items'][()])
                    else:
                        # List was stored as individual items
                        length = item.attrs.get('length', 0)
                        result = [self.load(f"{group_path}/item_{i}") for i in range(length)]

                    # Convert to tuple if needed
                    if item.attrs.get('type') == 'tuple':
                        result = tuple(result)

                    if load_metadata:
                        return result, metadata
                    return result

                # Check if this is a None value
                elif item.attrs.get('type') == 'none':
                    if load_metadata:
                        return None, metadata
                    return None

                # Default case for groups - treat as dict
                result = {}
                for key in item:
                    result[key] = self.load(f"{group_path}/{key}")

                if load_metadata:
                    return result, metadata
                return result

            # Handle datasets
            else:
                # Check if this is a pickled object
                if item.attrs.get('type') == 'pickled':
                    pickled_data = item[()].tobytes()
                    result = pickle.loads(pickled_data)
                    if load_metadata:
                        return result, metadata
                    return result

                # Get the data
                data = item[()]

                # Handle primitive types
                if item.attrs.get('type') in ('int', 'float', 'bool', 'str'):
                    if len(data) == 1:
                        result = data[0]
                        if item.attrs.get('type') == 'int':
                            result = int(result)
                        elif item.attrs.get('type') == 'float':
                            result = float(result)
                        elif item.attrs.get('type') == 'bool':
                            result = bool(result)
                    else:
                        result = data

                    if load_metadata:
                        return result, metadata
                    return result

                # Default case - return array directly
                if load_metadata:
                    return data, metadata
                return data

    def save_dataframe(self, df, group_path, metadata=None):
        """
        Save a pandas DataFrame to the HDF5 file.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to save
        group_path : str
            The path within the HDF5 file to save the DataFrame
        metadata : dict, optional
            Additional metadata to store with the DataFrame
        """
        self._ensure_parent_exists()

        with h5py.File(self.path, 'a') as hf:
            if group_path in hf:
                del hf[group_path]

            g = hf.create_group(group_path)

            # Save index and columns
            if df.index.name:
                g.attrs['index_name'] = df.index.name

            # Save values
            g.create_dataset('values', data=df.values, compression="gzip", compression_opts=9)

            # Save index
            index_values = df.index.values
            if isinstance(df.index, pd.MultiIndex):
                g.attrs['index_type'] = 'multi'
                for i, name in enumerate(df.index.names):
                    if name:
                        g.attrs[f'index_name_{i}'] = name
                # Save each level of the MultiIndex
                for i, level in enumerate(zip(*df.index.values)):
                    g.create_dataset(f'index_level_{i}', data=np.array(level))
            else:
                g.attrs['index_type'] = 'single'
                if isinstance(index_values[0], str):
                    dt = h5py.special_dtype(vlen=str)
                    g.create_dataset('df_index', data=np.array(index_values, dtype=dt))
                else:
                    g.create_dataset('df_index', data=index_values)

            # Save columns
            if isinstance(df.columns, pd.MultiIndex):
                g.attrs['columns_type'] = 'multi'
                for i, name in enumerate(df.columns.names):
                    if name:
                        g.attrs[f'columns_name_{i}'] = name
                # Save each level of the MultiIndex
                for i, level in enumerate(zip(*df.columns.values)):
                    g.create_dataset(f'columns_level_{i}', data=np.array(level, dtype=h5py.special_dtype(vlen=str)))
            else:
                g.attrs['columns_type'] = 'single'
                dt = h5py.special_dtype(vlen=str)
                g.create_dataset('df_columns', data=np.array(df.columns.values, dtype=dt))

            # Add metadata
            g.attrs['type'] = 'dataframe'
            if metadata:
                for key, value in metadata.items():
                    g.attrs[key] = value

    def load_dataframe(self, group_path):
        """
        Load a pandas DataFrame from the HDF5 file.

        Parameters
        ----------
        group_path : str
            The path within the HDF5 file where the DataFrame is stored

        Returns
        -------
        pandas.DataFrame
            The loaded DataFrame
        """
        if not self.path.exists():
            return None

        with h5py.File(self.path, 'r') as hf:
            if group_path not in hf:
                return None

            g = hf[group_path]

            # Load values
            values = g['values'][()]

            # Load index
            if g.attrs.get('index_type') == 'multi':
                # Reconstruct MultiIndex
                levels = []
                for i in range(len([k for k in g.keys() if k.startswith('index_level_')])):
                    levels.append(g[f'index_level_{i}'][()])

                # Get level names
                names = []
                for i in range(len(levels)):
                    name = g.attrs.get(f'index_name_{i}', None)
                    names.append(name)

                # Create MultiIndex
                index = pd.MultiIndex.from_arrays(levels, names=names)
            else:
                # Load regular Index
                index = g['df_index'][()]
                index_name = g.attrs.get('index_name', None)
                index = pd.Index(index, name=index_name)

            # Load columns
            if g.attrs.get('columns_type') == 'multi':
                # Reconstruct MultiIndex for columns
                levels = []
                for i in range(len([k for k in g.keys() if k.startswith('columns_level_')])):
                    levels.append(g[f'columns_level_{i}'][()])

                # Get level names
                names = []
                for i in range(len(levels)):
                    name = g.attrs.get(f'columns_name_{i}', None)
                    names.append(name)

                # Create MultiIndex
                columns = pd.MultiIndex.from_arrays(levels, names=names)
            else:
                # Load regular columns
                columns = g['df_columns'][()]

            # Create DataFrame
            return pd.DataFrame(values, index=index, columns=columns)

    def save_series(self, series, group_path, metadata=None):
        """
        Save a pandas Series to the HDF5 file.

        Parameters
        ----------
        series : pandas.Series
            The Series to save
        group_path : str
            The path within the HDF5 file to save the Series
        metadata : dict, optional
            Additional metadata to store with the Series
        """
        self._ensure_parent_exists()

        with h5py.File(self.path, 'a') as hf:
            if group_path in hf:
                del hf[group_path]

            g = hf.create_group(group_path)

            # Save name
            if series.name:
                g.attrs['name'] = series.name

            # Save values
            g.create_dataset('values', data=series.values, compression="gzip", compression_opts=9)

            # Save index
            index_values = series.index.values
            if isinstance(series.index, pd.MultiIndex):
                g.attrs['index_type'] = 'multi'
                for i, name in enumerate(series.index.names):
                    if name:
                        g.attrs[f'index_name_{i}'] = name
                # Save each level of the MultiIndex
                for i, level in enumerate(zip(*series.index.values)):
                    g.create_dataset(f'index_level_{i}', data=np.array(level))
            else:
                g.attrs['index_type'] = 'single'
                if series.index.name:
                    g.attrs['index_name'] = series.index.name

                if isinstance(index_values[0], str):
                    dt = h5py.special_dtype(vlen=str)
                    g.create_dataset('series_index', data=np.array(index_values, dtype=dt))
                else:
                    g.create_dataset('series_index', data=index_values)

            # Add metadata
            g.attrs['type'] = 'series'
            if metadata:
                for key, value in metadata.items():
                    g.attrs[key] = value

    def load_series(self, group_path):
        """
        Load a pandas Series from the HDF5 file.

        Parameters
        ----------
        group_path : str
            The path within the HDF5 file where the Series is stored

        Returns
        -------
        pandas.Series
            The loaded Series
        """
        if not self.path.exists():
            return None

        with h5py.File(self.path, 'r') as hf:
            if group_path not in hf:
                return None

            g = hf[group_path]

            # Load values
            values = g['values'][()]

            # Load name
            name = g.attrs.get('name', None)

            # Load index
            if g.attrs.get('index_type') == 'multi':
                # Reconstruct MultiIndex
                levels = []
                for i in range(len([k for k in g.keys() if k.startswith('index_level_')])):
                    levels.append(g[f'index_level_{i}'][()])

                # Get level names
                names = []
                for i in range(len(levels)):
                    name_i = g.attrs.get(f'index_name_{i}', None)
                    names.append(name_i)

                # Create MultiIndex
                index = pd.MultiIndex.from_arrays(levels, names=names)
            else:
                # Load regular Index
                index = g['series_index'][()]
                index_name = g.attrs.get('index_name', None)
                index = pd.Index(index, name=index_name)

            # Create Series
            return pd.Series(values, index=index, name=name)

    def list_groups(self):
        """
        List all available top-level groups in the HDF5 file.

        Returns
        -------
        list
            List of group names
        """
        if not self.path.exists():
            return []

        with h5py.File(self.path, 'r') as hf:
            return list(hf.keys())

    def save_result_comparisons(self, result_comparisons, dataset_name):
        """
        Save a ResultComparisonsType object to the HDF5 file.

        Parameters
        ----------
        result_comparisons : ResultComparisonsType
            The result comparisons object to save
        dataset_name : str
            Name of the dataset
        """
        # Create a group for this dataset
        group_path = f"/datasets/{dataset_name}"

        # Save outputs
        self.save(result_comparisons.outputs, f"{group_path}/outputs")

        # Save weight_strength_relation
        self.save(result_comparisons.weight_strength_relation, f"{group_path}/weight_strength_relation")

        # Save config as pickled object (maintaining all methods)
        with h5py.File(self.path, 'a') as hf:
            if f"{group_path}/config" in hf:
                del hf[f"{group_path}/config"]

            pickled_config = pickle.dumps(result_comparisons.config)
            dset = hf.create_dataset(f"{group_path}/config", data=np.void(pickled_config))
            dset.attrs['type'] = 'pickled'
            dset.attrs['class'] = type(result_comparisons.config).__name__

        # Save metadata
        with h5py.File(self.path, 'a') as hf:
            if group_path in hf:
                g = hf[group_path]
                g.attrs['dataset_name'] = dataset_name
                g.attrs['saved_at'] = datetime.now().isoformat()
                g.attrs['type'] = 'ResultComparisonsType'

    def load_result_comparisons(self, dataset_name):
        """
        Load a ResultComparisonsType object from the HDF5 file.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset

        Returns
        -------
        ResultComparisonsType
            The loaded result comparisons object
        """
        if not self.path.exists():
            return None

        group_path = f"/datasets/{dataset_name}"

        with h5py.File(self.path, 'r') as hf:
            if group_path not in hf:
                return None

        outputs = self.load(f"{group_path}/outputs")
        weight_strength_relation = self.load(f"{group_path}/weight_strength_relation")

        # Load config (pickle)
        with h5py.File(self.path, 'r') as hf:
            if f"{group_path}/config" in hf:
                pickled_data = hf[f"{group_path}/config"][()].tobytes()
                config = pickle.loads(pickled_data)
            else:
                config = None

        # Import here to avoid circular imports
        from crowd_certain.utilities.utils import ResultComparisonsType

        return ResultComparisonsType(
            outputs=outputs,
            config=config,
            weight_strength_relation=weight_strength_relation
        )

    def save_all_datasets_results(self, results_dict):
        """
        Save results for multiple datasets.

        Parameters
        ----------
        results_dict : Dict
            A dictionary mapping dataset names to their results
        """
        for dataset_name, result in results_dict.items():
            self.save_result_comparisons(result, str(dataset_name.value) if hasattr(dataset_name, 'value') else str(dataset_name))

    def load_all_datasets_results(self, dataset_names):
        """
        Load results for multiple datasets.

        Parameters
        ----------
        dataset_names : List
            A list of dataset names to load

        Returns
        -------
        Dict
            A dictionary mapping dataset names to their loaded results
        """
        results = {}
        for name in dataset_names:
            name_str = str(name.value) if hasattr(name, 'value') else str(name)
            result = self.load_result_comparisons(name_str)
            if result is not None:
                results[name] = result
        return results

    def get_metadata(self):
        """
        Get metadata about all datasets in the HDF5 file.

        Returns
        -------
        dict
            Dictionary containing metadata for each dataset
        """
        if not self.path.exists():
            return {}

        metadata = {}
        with h5py.File(self.path, 'r') as hf:
            if 'datasets' not in hf:
                return {}

            for dataset_name in hf['datasets']:
                dataset_group = hf[f'datasets/{dataset_name}']
                dataset_metadata = {}
                for key, value in dataset_group.attrs.items():
                    dataset_metadata[key] = value
                metadata[dataset_name] = dataset_metadata

        return metadata
