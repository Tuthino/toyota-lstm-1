import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from mylib import *
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout # type: ignore
from keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler

###### CONFIG ######
input_csv = 'csv/SingapurCargoVesselsMonthly.csv'
models_base_dir = 'models'
model_dir_tag = 'dropout'
model_dir = create_model_dir(models_base_dir, model_dir_tag)
best_model_path = f"{model_dir}/best_model.h5"


sequence_length = 18 # how many months to look back
cfg_epochs = 400
cfg_patience = 90
cfg_batch_size = 32
cfg_LSTM_units = 128
cfg_dropout = 0.3
cfg_train_ratio = 0.70
cfg_validation_ratio = 0.15
cfg_optimizer = 'adam'
cfg_loss = 'mean_squared_error'

run_params = {
    'input_csv': input_csv,
    'sequence_length': sequence_length,
    'cfg_train_ratio': cfg_train_ratio,
    'cfg_validation_ratio': cfg_validation_ratio,
    'cfg_epochs': cfg_epochs,
    'cfg_patience': cfg_patience,
    'cfg_batch_size': cfg_batch_size,
    'cfg_LSTM_units': cfg_LSTM_units,
    'cfg_dropout': cfg_dropout,
    'cfg_optimizer': cfg_optimizer,
    'cfg_loss': cfg_loss,
}

###### END OF CONFIG ######




## Load and preprocess / show the dataset
# Load the dataset
dataset = pd.read_csv(input_csv)
dataset['month'] = pd.to_datetime(dataset['month'], format='%Y-%m')
# print(dataset.head())

# TODO:
# Perform data preprocessing steps (e.g., handle missing values, normalization)

# Show the diagram of original data for specific months
# filtered_data = filter_data(dataset,'1993-01', '1993-12')
# show_diagram(filtered_data)


# Replace 'vessels.csv' with your actual CSV file path
df = pd.read_csv(input_csv, parse_dates=['month'])

# Make sure the dataframe is sorted by date
df = df.sort_values(by='month').reset_index(drop=True)

# Extract values (monthly vessel counts)
vessel_data = df['vessels'].values.reshape(-1, 1)

# Scale the data to [0,1] range to help LSTM training
scaler = MinMaxScaler(feature_range=(0, 1))
vessel_data_scaled = scaler.fit_transform(vessel_data)

dataset_size = len(df)
# Prepare the input and output sequences
X_all, y_all = create_sequences(vessel_data_scaled, sequence_length)

# Train-test split
# Split the data into training and testing sets 80/20
# For example, use the last 12 months (or more) as test
train_size = int(len(X_all) * cfg_train_ratio)
val_end= int(len(X_all) * (cfg_train_ratio + cfg_validation_ratio))

X_train, y_train = X_all[:train_size], y_all[:train_size]
X_val, y_val     = X_all[train_size:val_end], y_all[train_size:val_end]
X_test, y_test   = X_all[val_end:], y_all[val_end:]

# -------------------------------------------------
# 2. Build the LSTM Model
# -------------------------------------------------

# Create checkopoint callback,
# to save only the best model 

checkpoint = ModelCheckpoint(
    best_model_path,         # TODO: add path, and create unique model names
    monitor='val_loss',      # metric to monitor
    verbose=1,               # verbosity mode (1 = progress messages)
    save_best_only=True,     # only save when the monitored metric improves
    mode='min'               # mode should be 'min' if you're monitoring loss
)

# Callback to stop training if validation loss doesn't improve for X epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=cfg_patience,
    verbose=1,
    restore_best_weights=True
)


# TODO add Dropout layers to prevent overfitting
model = Sequential()
model.add(LSTM(cfg_LSTM_units, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(cfg_dropout))  # Randomly drops X% of the outputs from the LSTM layer
model.add(LSTM(cfg_LSTM_units))
model.add(Dropout(cfg_dropout))  # Randomly drops X% of the outputs from the LSTM layer
model.add(Dense(1))

model.compile(optimizer=cfg_optimizer, loss=cfg_loss)

# -------------------------------------------------
# 3. Train the Model
# -------------------------------------------------
history = model.fit(
    X_train, y_train, 
    epochs=cfg_epochs, 
    batch_size=cfg_batch_size,
    validation_data=(X_val, y_val), 
    verbose=1,
    callbacks=[checkpoint,early_stopping]  # Add the checkpoint callback
    )

# -------------------------------------------------
# 4. Evaluate & Predict
# -------------------------------------------------
# Load the best model
best_model = load_model(best_model_path)



# Predictions on test set
predictions_scaled = best_model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)
actual = scaler.inverse_transform(y_test)



# Saving model config and predictions diagram
print(f"Num of predictions: {len(predictions)}")
print(f"Num of actual test entries: {len(actual)}")

# The evaluation on test set is done within the model.fit(),
# So we just read from the history object
save_model_loss(model_dir, min(history.history['val_loss']))


save_model_config(best_model, run_params, model_dir)


# Plot test predictions vs. actual values
test_months = df['month'].iloc[val_end + sequence_length:].reset_index(drop=True)

show_predictions_diagram(
    test_months, 
    predictions, 
    actual, 
    save_path=f"{model_dir}/predictions_diagram.png"
)
