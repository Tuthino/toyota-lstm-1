import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylib import *
## For Keras backend we are using the JAX
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

## Load and preprocess / show the dataset
# Load the dataset
input_csv = 'csv/SingapurCargoVesselsMonthly.csv'
dataset = pd.read_csv(input_csv)
dataset['month'] = pd.to_datetime(dataset['month'], format='%Y-%m')
print(dataset.head())

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

# (B) Scale the data to [0,1] range to help LSTM training
scaler = MinMaxScaler(feature_range=(0, 1))
vessel_data_scaled = scaler.fit_transform(vessel_data)

dataset_size = len(df)
# Prepare the input and output sequences
sequence_length = 6 # how many months to look back
X_all, y_all = create_sequences(vessel_data_scaled, sequence_length)

# (D) Train-test split
# Split the data into training and testing sets 80/20
# For example, use the last 12 months (or more) as test
train_size = int(len(X_all) * 0.8)
X_train, y_train = X_all[:train_size], y_all[:train_size]
X_test, y_test = X_all[train_size:], y_all[train_size:]

print(f"x_train {X_train} y_train {y_train}")

# -------------------------------------------------
# 2. Build the LSTM Model
# -------------------------------------------------

# Create checkopoint callback,
# to save only the best model 

checkpoint = ModelCheckpoint(
    'models/best_model.h5',         # TODO: add path, and create unique model names
    monitor='val_loss',      # metric to monitor
    verbose=1,               # verbosity mode (1 = progress messages)
    save_best_only=True,     # only save when the monitored metric improves
    mode='min'               # mode should be 'min' if you're monitoring loss
)

# TODO add Dropout layers to prevent overfitting
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# -------------------------------------------------
# 3. Train the Model
# -------------------------------------------------
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=16,
    validation_data=(X_test, y_test), 
    verbose=1,
    callbacks=[checkpoint]  # Add the checkpoint callback
    )

# -------------------------------------------------
# 4. Evaluate & Predict
# -------------------------------------------------
# Load the best model
best_model = load_model('models/best_model.h5')



# (A) Predictions on test set
predictions_scaled = best_model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)
actual = scaler.inverse_transform(y_test)


# (B) Plot test predictions vs. actual values
plt.figure()
plt.plot(range(len(actual)), actual, label='Actual')
plt.plot(range(len(predictions)), predictions, label='Predicted')
plt.title('Test Set: Actual vs. Predicted Vessel Counts')
plt.xlabel('Test Samples')
plt.ylabel('Vessel Count')
plt.legend()
plt.show()

