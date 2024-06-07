# import libraries
import os
import time
import pickle
import pandas as pd
import pyarrow.parquet as pa
from sklearn.feature_extraction import DictVectorizer

import warnings
warnings.filterwarnings('ignore')

# setting path to the data directory
CURRENT_DIRECTORY = os.getcwd()
PARENT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
DATA_PATH = os.path.join(PARENT_DIRECTORY, '_data')

start_time = time.time()

'''
    Notes:
    1. We Shall use the code the code of Data Pre-processing written for Week-01
    2. Here we are using Yellow taxi Data of January, February, March Months
    3. train=> January, validation=>February, test=March
'''
# setting path to the data directory
CURRENT_DIRECTORY = os.getcwd()
PARENT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
DATA_PATH = os.path.join(PARENT_DIRECTORY, '_data')
PICKLE_PATH = os.path.join(CURRENT_DIRECTORY, '__pickle')

dest_path = os.getcwd()

vectorise = DictVectorizer()


# Join the path
def path_join(train, val, test):
    # Join January data path
    train_data_path = os.path.join(DATA_PATH, train)
    # Join February data path
    val_data_path = os.path.join(DATA_PATH, val)
    # Join March data path
    test_data_path = os.path.join(DATA_PATH, test)

    return [train_data_path, val_data_path, test_data_path]


# read the data
def read_data(data):
    if data.endswith('.parquet'):
        data = pa.read_table(data)
        df = data.to_pandas() # converting to pandas df
        df.columns = df.columns.str.lower()
        return df
    elif data.endswith('.csv'):
        df = pd.read_csv(data)
        df.columns = df.columns.str.lower()
        return df

    else:
        return 'Not valid format'


# To calculate the standard deviation of the pick and drop time in minutes

def pre_processing(data, choice):
    data['duration'] = pd.to_datetime(data['lpep_dropoff_datetime']) - pd.to_datetime(data['lpep_pickup_datetime'])
    # Convert duration to total seconds
    data['duration'] = data['duration'].dt.total_seconds()
    # Convert seconds to hours and minutes
    data['duration'] = data['duration'] / 60

    # Identifying the outliers
    data_outliers = data[(data['duration']>=1)&(data['duration']<=60)]

    # Converting pick up and drop off location id into strings
    data_outliers['pulocationid'] = data_outliers['pulocationid'].astype(str)
    data_outliers['dolocationid'] = data_outliers['dolocationid'].astype(str)

    # Converting DataFrame into a list of dictionaries
    df_dict = data_outliers[['pulocationid', 'dolocationid', 'trip_distance']].to_dict(orient='records')

    if choice == 0:
        X_train = vectorise.fit_transform(df_dict)
        return X_train
    elif choice == 1:
        X_val = vectorise.transform(df_dict)
        return X_val
    else:
        return 'Enter Choice 0 or 1'


def save_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def main(train, val, test):
    data_path_files = path_join(train, val, test)

    df_train = read_data(data_path_files[0]) # READ JANUARY DATA

    df_val = read_data(data_path_files[1]) # READ FEBRUARY DATA

    df_test = read_data(data_path_files[2]) # READ MARCH DATA

    y_train = df_train['duration']
    y_val = df_val['duration']
    y_test = df_test['duration']

    X_train = pre_processing(df_train, choice=0)
    X_val = pre_processing(df_val, choice=1)
    X_test = pre_processing(df_test, choice=1)

    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    save_pickle(vectorise, os.path.join(dest_path, "vectorise.pkl"))
    save_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    save_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    save_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

if __name__ == '__main__':
    # File Name
    january_file_name = 'green_tripdata_2023-01.parquet'
    february_file_name = 'green_tripdata_2023-02.parquet'
    march_file_name = 'green_tripdata_2023-03.parquet'

    main(january_file_name, february_file_name, march_file_name)
