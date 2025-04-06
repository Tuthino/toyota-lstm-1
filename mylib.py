# Helper functions for the main script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)



# def create_sequences(data, sequence_length):
#     # Create sequences of data for LSTM
#     # X holds and array of sequences
#     # y hold the next value in the sequence
#     values = data['vessels'].values.astype(int)
#     X, y = [], []
#     for i in range(len(data) - sequence_length):
#         X.append(values[i:i+sequence_length])
#         y.append(values[i+sequence_length])
#     return np.array(X).reshape(-1, sequence_length, 1), np.array(y).reshape(-1, 1)

def show_diagram(dataset):
    # Show the diagram of original data
    plt.figure(figsize=(12, 6))
    plt.plot(dataset['month'], dataset['vessels'], marker='o')
    plt.xticks(dataset['month'], dataset['month'].dt.strftime('%Y-%m'), rotation=45)
    plt.title("Monthly Vessel Count")
    plt.xlabel("Month")
    plt.ylabel("Vessels")
    plt.grid(True)
    plt.show()

def filter_data(dataset, start, end):
    # Takes year-month strings as input, converts to pd compatible format and returns only filtered data
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    filtered_data = dataset.loc[(dataset['month'] >= start) & (dataset['month'] <= end)]
    return filtered_data


