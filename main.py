import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from mylib import *
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU # type: ignore
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler
from sklearn.model_selection import TimeSeriesSplit


###### CONFIG ######
input_csv = 'csv/SingapurCargoVesselsMonthly.csv'
models_base_dir = 'models'
model_dir_tag = 'dropout'
model_dir = create_model_dir(models_base_dir, model_dir_tag)
best_model_path = f"{model_dir}/best_model.h5"


sequence_length = 12 # how many months to look back
cfg_epochs = 700
cfg_patience = 80
cfg_batch_size = 128
cfg_LSTM_units = 64
cfg_dropout = 0.40
holdout_fraction = 0.10
cfg_folds_num = 10
cfg_learning_rate = 0.0005 *(cfg_batch_size   /4 ) # Adjust learning rate based on batch size
cfg_recurrent_dropout = 0.0
cfg_loss = 'mean_squared_error'

run_params = {
    'input_csv': input_csv,
    'sequence_length': sequence_length,
    'cfg_epochs': cfg_epochs,
    'cfg_patience': cfg_patience,
    'cfg_batch_size': cfg_batch_size,
    'cfg_LSTM_units': cfg_LSTM_units,
    'cfg_dropout': cfg_dropout,
    'holdout_fraction': holdout_fraction,
    'cfg_folds_num': cfg_folds_num,
    'cfg_learning_rate': cfg_learning_rate,
    'cfg_loss': cfg_loss,
}

###### END OF CONFIG ######




## Load and preprocess / show the dataset
# Load the dataset
dataset = pd.read_csv(input_csv)
dataset['month'] = pd.to_datetime(dataset['month'], format='%Y-%m')

# Show the diagram of original data for specific months
# filtered_data = filter_data(dataset,'1993-01', '1993-12')
# show_diagram(dataset)


# Replace 'vessels.csv' with your actual CSV file path
df = pd.read_csv(input_csv, parse_dates=['month'])


df['month_num'] = df['month'].dt.month  # Month as an integer from 1 to 12
df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

# Extract values (monthly vessel counts)
vessel_data = df['vessels'].values.reshape(-1, 1)

# Scale the data to [0,1] range to help LSTM training
scaler = MinMaxScaler(feature_range=(0, 1))

vessel_data_scaled = scaler.fit_transform(vessel_data)
# Extract cyclical features (already between -1 and 1)
month_features = df[['month_sin', 'month_cos']].values

# Combine features: each row now has [scaled vessels, month_sin, month_cos]
features = np.concatenate([vessel_data_scaled, month_features], axis=1)

# Create sequences using your helper function.
# Note: target_idx=0 means the target is the vessel count.
X_all, y_all = create_sequences(features, sequence_length, target_idx=0)
print(f"Total sequences: {len(X_all)}")

# Split the data into training+validation and test sets
# The holdout set is not visible for model during the training process
# It is to test the final performance on the data that it has never seen
holdout_size = int(len(X_all) * holdout_fraction)
X_trainval = X_all[:-holdout_size]
y_trainval = y_all[:-holdout_size]
X_test = X_all[-holdout_size:]
y_test = y_all[-holdout_size:]
print(f"Train+Val size for CV: {len(X_trainval)}")
print(f"Test size (hold-out): {len(X_test)}")

print(f"Total sequences: {len(X_all)}")

tscv = TimeSeriesSplit(n_splits=cfg_folds_num)  

cv_losses = []
fold = 1
best_fold_loss = float('inf')

# Loop over each fold
for train_index, val_index in tscv.split(X_all):
    print(f"\n====== Fold {fold} ======")
    X_train_cv, y_train_cv = X_all[train_index], y_all[train_index]
    X_val_cv, y_val_cv     = X_all[val_index], y_all[val_index]
    fold_model_path = f"{model_dir}/fold_{fold}_best_model.h5"

    
    # Create tf.data Datasets for this fold
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_cv, y_train_cv))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train_cv), reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(cfg_batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_cv, y_val_cv))
    val_dataset = val_dataset.batch(cfg_batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Define model architecture for this fold
    model = Sequential()
    # First GRU layer with input_shape; only need to specify input_shape here
    model.add(GRU(cfg_LSTM_units, return_sequences=True, 
                  input_shape=(sequence_length, 3),
                  recurrent_dropout=cfg_recurrent_dropout))
    # Second GRU layer returning final state only
    model.add(GRU(cfg_LSTM_units, return_sequences=False, 
                  recurrent_dropout=cfg_recurrent_dropout))
    model.add(Dropout(cfg_dropout))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=cfg_learning_rate)
    model.compile(optimizer=optimizer, loss=cfg_loss)
    
    # Setup callbacks for this fold 
    checkpoint = ModelCheckpoint(
    fold_model_path,         # TODO: add path, and create unique model names
    monitor='val_loss',      # metric to monitor
    verbose=1,               # verbosity mode (1 = progress messages)
    save_best_only=True,     # only save when the monitored metric improves
    mode='min'               # mode should be 'min' if you're monitoring loss
)
    early_stopping = EarlyStopping(monitor='val_loss', patience=cfg_patience, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, verbose=1, min_lr=1e-4)
    
    # Train the model for this fold
    history = model.fit(
        train_dataset,
        epochs=cfg_epochs,
        validation_data=val_dataset,
        callbacks=[early_stopping, reduce_lr,checkpoint],
        verbose=1
    )
    
    # Evaluate on the validation set for this fold
    fold_loss = model.evaluate(val_dataset, verbose=0)
    print(f"Fold {fold} validation loss: {fold_loss}")
    cv_losses.append(fold_loss)
    
    val_loss = model.evaluate(val_dataset, verbose=0)
    print(f"Fold {fold} validation loss: {val_loss}")
    cv_losses.append(val_loss)
    
    if val_loss < best_fold_loss:
        best_fold_loss = val_loss
        best_fold_model_path = fold_model_path
    fold += 1


print(f"\nAverage Cross-Validation Loss: {np.mean(cv_losses)}")
print(f"Best Fold Validation Loss: {best_fold_loss}")

# Final evaluation of best model on the holdout set
best_model = load_model(best_fold_model_path)
print("Loaded best model from cross-validation.")
save_model_loss(model_dir, min(history.history['val_loss']),best_fold_model_path)

save_model_config(best_model, run_params, model_dir)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(cfg_batch_size).prefetch(tf.data.AUTOTUNE)
test_loss = best_model.evaluate(test_dataset, verbose=0)
print(f"Test Loss of the best model: {test_loss}")

# Plotting the predictions of holdout set
predictions_scaled = best_model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))
test_months = df['month'].iloc[-len(X_test):].reset_index(drop=True)
show_predictions_diagram(test_months, predictions, actual, save_path=f"{model_dir}/predictions_diagram.png")
