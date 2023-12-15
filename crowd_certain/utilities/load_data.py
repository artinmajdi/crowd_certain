import pandas as pd
import numpy as np
import wget
from sklearn import preprocessing
from typing import *
from crowd_certain.utilities.params import ReadMode, DatasetNames

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def aim1_3_read_download_UCI_database(config, dataset_name=''):

    dataset_path = config.dataset.path_all_datasets
    dataset_name = dataset_name or config.dataset.dataset_name
    main_url     = config.dataset.main_url

    # dataset_path = pathlib.Path( dataset_path ).absolute()
    dataset_path.mkdir( parents=True, exist_ok=True )

    names, files, url = [], [], ''
    def read_raw_names_files():
        nonlocal names, files, url

        # main_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

        if dataset_name is DatasetNames.KR_VS_KP:
            names   = [f'a{i}' for i in range(36)] + ['true']
            files   = ['Index', f'{dataset_name}.data', f'{dataset_name}.names']
            url     = main_url + '/chess/king-rook-vs-king-pawn/'

        elif dataset_name is DatasetNames.MUSHROOM:
            names   = ['true'] + [f'a{i}' for i in range(22)]
            files = ['Index', f'{dataset_name}.data', f'{dataset_name}.names']
            url     = main_url + '/mushroom/'

        elif dataset_name is DatasetNames.SICK:
            names   = [f'a{i}' for i in range(29)] + ['true']
            files   = [f'{dataset_name}.data', f'{dataset_name}.names', f'{dataset_name}.test']
            url     = main_url + '/thyroid-disease/'

        elif dataset_name is DatasetNames.SPAMBASE:
            names   = [f'a{i}' for i in range(57)] + ['true']
            files   = [f'{dataset_name}.DOCUMENTATION', f'{dataset_name}.data', f'{dataset_name}.names', f'{dataset_name}.zip']
            url     = main_url + '/spambase/'

        elif dataset_name is DatasetNames.TIC_TAC_TOE:
            names   = [f'a{i}' for i in range(9)] + ['true']
            files   = [f'{dataset_name}.data', f'{dataset_name}.names']
            url     = main_url + '/tic-tac-toe/'

        elif dataset_name is DatasetNames.WAVEFORM:
            names   = [f'a{i}' for i in range(21)] + ['true']
            files   = [ 'Index', f'{dataset_name}-+noise.c', f'{dataset_name}-+noise.data.Z', f'{dataset_name}-+noise.names', f'{dataset_name}.c', f'{dataset_name}.data.Z', f'{dataset_name}.names']
            url     = main_url + '/mwaveform/'

        # elif dataset_name is DatasetNames.BIODEG:
        #     names   = [f'a{i}' for i in range(41)] + ['true']
        #     files   = [f'{dataset_name}.csv']
        #     url     = main_url + '/00254/'

        # elif dataset_name is DatasetNames.HORSE_COLIC:
        #     names   = [f'a{i}' for i in range(41)] + ['true']
        #     files   = [f'{dataset_name}.data', f'{dataset_name}.names', f'{dataset_name}.names.original', f'{dataset_name}.test']
        #     url     = main_url + '/horse-colic/'

        elif dataset_name is DatasetNames.IONOSPHERE:
            names   = [f'a{i}' for i in range(34)] + ['true']
            files   = [ 'Index', f'{dataset_name}.data', f'{dataset_name}.names']
            url     = main_url + '/ionosphere/'

    read_raw_names_files()

    def download_data():
        local_path = dataset_path.joinpath( f'UCI_{dataset_name}' )
        local_path.mkdir(exist_ok=True)

        for name in files:
            wget.download(url + name, local_path)

        data_raw = pd.read_csv(local_path.joinpath(f'{dataset_name}.data'))
        return data_raw, []

    def separate_train_test(data_raw, train_frac=0.8) -> Dict[str, pd.DataFrame]:
        train = data_raw.sample(frac=train_frac, random_state=config.dataset.random_state).sort_index()
        test = data_raw.drop(train.index)
        return dict(train=train, test=test)

    def reading_from_arff():

        def do_read():
            def read_data_after_at_data_line():
                dir_dataset = dataset_path / f'{dataset_name}/{dataset_name}.arff'
                with open(dir_dataset, 'r') as f:
                    for line in f:
                        if line.lower().startswith('@data'):
                            break
                    return pd.read_csv(f, header=None, sep=',', na_values=['?','nan','null','NaN','NULL'])

            def changing_str_to_int():
                le = preprocessing.LabelEncoder()
                for name in data_raw.columns:
                    if data_raw[name].dtype == 'object':
                        data_raw[name] = le.fit_transform(data_raw[name])

            data_raw = read_data_after_at_data_line()

            changing_str_to_int()

            feature_columns = [f'a{i}' for i in range(data_raw.shape[1]-1)]
            data_raw.columns = feature_columns + ['true']

            data_raw.replace( 2147483648, np.nan, inplace=True)
            data_raw.replace(-2147483648, np.nan, inplace=True)

            # removing columns that only has one value (mostly the one that are fully NaN)
            for name in data_raw.columns:
                if len(data_raw[name].unique()) == 1:
                    data_raw.drop(columns=name, inplace=True)
                    feature_columns.remove(name)

            # extracting only classes "1" and "2" to correspond to Tao et al. paper
            if dataset_name is DatasetNames.WAVEFORM:
                data_raw = data_raw[data_raw.true != 0]
                data_raw.true.replace({1: 0, 2: 1}, inplace=True)

            if dataset_name is DatasetNames.SICK:
                data_raw.replace({np.nan:0}, inplace=True)

            # if dataset_name is DatasetNames.BALANCE_SCALE:
            #     data_raw = data_raw[data_raw.true != 1]
            #     data_raw.true.replace({2:1}, inplace=True)

            if dataset_name is DatasetNames.IRIS:
                data_raw = data_raw[data_raw.true != 2]

            if dataset_name is DatasetNames.CAR:
                data_raw.true.replace({2:1,3:1}, inplace=True) # classes are [unacceptable, acceptable, good, very good]

            return data_raw, feature_columns

        data_raw, feature_columns = do_read()
        data_ = separate_train_test(data_raw=data_raw, train_frac=0.8)
        return data_, feature_columns

    def read_data():
        def postprocess(data_raw, names):

            def replacing_classes_char_to_int():
                # finding the unique classes
                lbls = set()
                for fx in feature_columns:
                    lbls = lbls.union(data_raw[fx].unique())

                # replacing the classes from char to int
                for ix, lb in enumerate(lbls):
                    data_raw[feature_columns] = data_raw[feature_columns].replace(lb,ix+1)

            feature_columns = names.copy()
            feature_columns.remove('true')

            if dataset_name is DatasetNames.KR_VS_KP:

                # changing the true labels from string to [0,1]
                data_raw.true = data_raw.true.replace('won',1).replace('nowin',0)

                # replacing the classes from char to int
                replacing_classes_char_to_int()

            elif dataset_name is DatasetNames.MUSHROOM:

                # changing the true labels from string to [0,1]
                data_raw.true = data_raw.true.replace('e',1).replace('p',0)

                # feature a10 has missing data
                data_raw.drop(columns=['a10'], inplace=True)
                feature_columns.remove('a10')

                # replacing the classes from char to int
                replacing_classes_char_to_int()

            elif dataset_name is DatasetNames.SICK:
                data_raw.true = data_raw.true.map(lambda x: x.split('.')[0]).replace('sick',1).replace('negative',0)
                column_name = 'a27' # 'TBG measured'
                data_raw = data_raw.drop(columns=[column_name])
                feature_columns.remove(column_name)

                # replacing the classes from char to int
                # data_raw = replacing_classes_char_to_int(data_raw, feature_columns)

            elif dataset_name is DatasetNames.SPAMBASE:
                pass

            elif dataset_name is DatasetNames.TIC_TAC_TOE:
                # renaming the two classes "good" and "bad" to "0" and "1"
                data_raw.true = data_raw.true.replace('negative',0).replace('positive',1)
                data_raw[feature_columns] = data_raw[feature_columns].replace('thresh_technique',1).replace('o',2).replace('b',0)

            # elif dataset_name is DatasetNames.SPLICE:
            #     pass

            # elif dataset_name is DatasetNames.THYROID:
            #     pass

            elif dataset_name is DatasetNames.WAVEFORM:
                # extracting only classes "1" and "2" to correspond to Tao et al. paper
                class_0 = data_raw[data_raw.true == 0].index
                data_raw.drop(class_0, inplace=True)
                data_raw.true = data_raw.true.replace(1,0).replace(2,1)

            # elif dataset_name is DatasetNames.BIODEG:
            #     data_raw.true = data_raw.true.replace('RB',1).replace('NRB',0)

            # elif dataset_name is DatasetNames.HORSE_COLIC:
            #     pass

            elif dataset_name is DatasetNames.IONOSPHERE:
                data_raw.true = data_raw.true.replace('g',1).replace('b',0)

            elif dataset_name in (12, 'vote'):
                pass

            return feature_columns

        filepath= dataset_path / f'UCI_{dataset_name}/{dataset_name}.data'

        filepath = filepath.with_suffix('.csv') if dataset_name == 'biodeg' else filepath
        delimiter = {'biodeg':';' , 'horse-colic':' '}.get(dataset_name)
        params = {'delimiter': delimiter}

        if config.dataset.read_mode == ReadMode.READ:
            params.update({'names': names})

        data_raw = pd.read_csv(filepath, **params)
        feature_columns = postprocess( data_raw, names) if config.dataset.read_mode == ReadMode.READ else []

        data = separate_train_test(
            data_raw=data_raw, train_frac=config.dataset.train_test_ratio)

        return data, feature_columns

    if   config.dataset.read_mode is ReadMode.READ_ARFF: return reading_from_arff()
    elif config.dataset.read_mode is ReadMode.READ:      return read_data()
    elif config.dataset.read_mode is ReadMode.DOWNLOAD:  return download_data()



