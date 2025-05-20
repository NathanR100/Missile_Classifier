########## Model Creation ###################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sequence_length = 1192
num_features = 5

model = Sequential()
model.add(Input(shape=(sequence_length, num_features)))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# This LSTM layer also returns sequences
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# Wrap the Dense layer in TimeDistributed to apply it to each time step
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['recall','precision','accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('my_model.keras', save_best_only=True)
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    # sample_weight=sample_weights,  # Use sample_weight instead of class_weight
    callbacks=callbacks
)
