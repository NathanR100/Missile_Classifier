#importing data

import pandas as pd

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Drop the 'sensor_id' column from df_train by specifying axis=1
df_train.drop('sensor_id', axis=1, inplace=True)
# Drop the 'sensor_id' column from df_test, axis=1 is the default for drop
df_test.drop('sensor_id', axis=1, inplace=True)

##Converting timestamp values to unix time then creating a relative timeframe per track

df_train = df_train.sort_values(by=['track_id', 'timestamp']).reset_index(drop=True)

df_train['timestamp'] = pd.to_datetime(df_train['timestamp'], unit='s')

df_train['timestamp'] = df_train.groupby('track_id')['timestamp'].transform(
    lambda x: (x - x.min()).dt.total_seconds()
)

##Scaling features per track_id using MinMaxScaler to preserve information from negative values

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

features_to_scale = ['latitude', 'longitude', 'altitude', 'radiometric_intensity']
df_train[features_to_scale] = df_train.groupby('track_id')[features_to_scale].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
)

## Time Series Window Per Track_ID
# each track_id has a ~614-1192 timepoints in 0.5 s intervals
# to preserve all information we will have a standard time window of 1192

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Split and pad sequences (best to split then pad)
# Get unique track IDs
track_ids = df_train['track_id'].unique()

# Split track IDs into train and validation
train_ids, val_ids = train_test_split(track_ids, test_size=0.2, random_state=42)

# Filter sequences and labels by these IDs
def filter_by_ids(df, ids):
    return df[df['track_id'].isin(ids)]

df_train_split = filter_by_ids(df_train, train_ids)
df_val_split = filter_by_ids(df_train, val_ids)

# Now recreate padded sequences and labels for train and val sets separately

def create_padded_sequences(df):

    sequences = df.groupby('track_id').apply(
        lambda g: g[['timestamp', 'latitude', 'longitude', 'altitude', 'radiometric_intensity']].values
    ).tolist()

    label_sequences = df.groupby('track_id')['reentry_phase'].apply(list).tolist()

    X = pad_sequences(sequences, maxlen=1192, dtype='float32', padding='post')
    y = pad_sequences(label_sequences, maxlen=1192, dtype='float32', padding='post')

    return X, y

X_train, y_train = create_padded_sequences(df_train_split)
X_val, y_val = create_padded_sequences(df_val_split)
