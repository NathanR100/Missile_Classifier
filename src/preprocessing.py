import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

# Load data
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Drop sensor_id column
df_train.drop('sensor_id', axis=1, inplace=True)
df_test.drop('sensor_id', axis=1, inplace=True)

# Features to scale
features_to_scale = ['latitude', 'longitude', 'altitude', 'radiometric_intensity']

# === Shared Utility Functions ===
def convert_timestamp(df):
    """
    Convert unix timestamp to relative seconds per track_id.
    """
    df = df.sort_values(by=['track_id', 'timestamp']).reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['timestamp'] = df.groupby('track_id')['timestamp'].transform(
        lambda x: (x - x.min()).dt.total_seconds()
    )
    return df

def normalize_features(df, features):
    """
    Apply MinMax scaling per track_id for specified features.
    """
    df[features] = df.groupby('track_id')[features].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )
    return df

def create_padded_sequences(df):
    """
    Create padded input and label sequences for training/validation.
    """
    sequences = df.groupby('track_id').apply(
        lambda g: g[['timestamp', 'latitude', 'longitude', 'altitude', 'radiometric_intensity']].values
    ).tolist()

    label_sequences = df.groupby('track_id')['reentry_phase'].apply(list).tolist()

    X = pad_sequences(sequences, maxlen=1192, dtype='float32', padding='post')
    y = pad_sequences(label_sequences, maxlen=1192, dtype='float32', padding='post')

    return X, y

def create_padded_test_sequences(df):
    """
    Create padded input sequences for test data (no labels).
    """
    sequences = df.groupby('track_id').apply(
        lambda g: g[['timestamp', 'latitude', 'longitude', 'altitude', 'radiometric_intensity']].values
    ).tolist()

    X = pad_sequences(sequences, maxlen=1192, dtype='float32', padding='post')
    return X


# === Preprocessing Pipeline ===

# Process train data
df_train = convert_timestamp(df_train)
df_train = normalize_features(df_train, features_to_scale)

# Split train into train/validation by track_id
track_ids = df_train['track_id'].unique()
train_ids, val_ids = train_test_split(track_ids, test_size=0.2, random_state=42)

def filter_by_ids(df, ids):
    return df[df['track_id'].isin(ids)]

df_train_split = filter_by_ids(df_train, train_ids)
df_val_split = filter_by_ids(df_train, val_ids)

X_train, y_train = create_padded_sequences(df_train_split)
X_val, y_val = create_padded_sequences(df_val_split)

# Process test data
df_test = convert_timestamp(df_test)
df_test = normalize_features(df_test, features_to_scale)
X_test = create_padded_test_sequences(df_test)

# === Now you have:
# X_train, y_train, X_val, y_val ready for training/validation
# X_test ready for inference

# === Saving X_test,X_val,y_val to my models folder to be used in evaluate.py
# Assuming X_test is your preprocessed test data (numpy array)
np.save("models/X_test.npy", X_test)
np.save("models/X_val.npy", X_val)
np.save("models/y_val.npy", y_val)
