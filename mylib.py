# Helper functions for the main script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import io
from contextlib import redirect_stdout

# def create_sequences(data, seq_length):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i : i + seq_length])
#         y.append(data[i + seq_length])
#      return np.array(X), np.array(y)
def create_sequences(data, seq_length, target_idx=0):
    """
    Create sequences of multi-dimensional features and select a target for each sequence.
    
    Parameters:
      data: np.array of shape (n_timesteps, n_features)
      seq_length: number of time steps to use for each input sequence
      target_idx: index of the target feature (default 0, assuming vessel count is the first column)
    
    Returns:
      X: array of input sequences with shape (num_samples, seq_length, n_features)
      y: array of targets with shape (num_samples, )
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, target_idx])
    return np.array(X), np.array(y)


def create_model_dir(base_dir, model_tag):
    # Create a directory for the model if it doesn't exist
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    model_dir = f"{base_dir}/{model_tag}_{timestamp}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir
# def show_predictions_diagram(actual_months, actual, pred_months, predictions, save_path=None):
#     # Plot full actual series and predicted series (which is shorter)
#     plt.figure(figsize=(12,6))
#     plt.plot(actual_months, actual, label='Actual', marker='o')
#     plt.plot(pred_months, predictions, label='Predicted', linestyle='--', marker='o')
#     plt.xlabel('Month')
#     plt.ylabel('Vessel Count')
#     plt.title('Test Data: Actual vs. Predicted Vessel Counts')
#     plt.legend()
#     plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
#     if save_path:
#         plt.savefig(save_path)
#         print(f"Plot saved to {save_path}")
#     plt.show()
def show_predictions_diagram(test_months, predictions, actual, save_path=None):
    # Show the diagram of original data and predictions
    plt.figure(figsize=(12,6))
    plt.plot(test_months, actual, label='Actual', marker='o')
    plt.plot(test_months, predictions, label='Predicted', linestyle='--', marker='o')
    plt.xlabel('Month')
    plt.ylabel('Vessel Count')
    plt.title('Test Data: Actual vs. Predicted Vessel Counts')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

def get_model_summary(model):
    summary_stream = io.StringIO()
    with redirect_stdout(summary_stream):
        model.summary()
    summary_string = summary_stream.getvalue()
    return summary_string

def save_model_config(model, run_params, model_dir):

    with open(f"{model_dir}/run_params.json", "w") as file:
        file.write(json.dumps(run_params, indent=4))
        print(f"Run parameters saved to {model_dir}/run_params.json")

    with open(f"{model_dir}/model_summary.txt", "w") as file:
        file.write(get_model_summary(model))
        print(f"Model summary saved to {model_dir}/model_sumarry.txt")

    with open(f"{model_dir}/model_config.json", "w") as file:
        file.write(json.dumps(model.get_config(), indent=4))
        print(f"Model configuration saved to {model_dir}/model_config.json")

    with open(f"{model_dir}/model_config_optimizer.json", "w") as file:
        file.write(json.dumps(model.get_config(), indent=4))
        print(f"Optimizer configuration saved to {model_dir}/model_config_optimizer.json")
        
def save_model_loss(model_dir, val_loss):
    # Save the model weights to a file
    # Currently we save only the val_loss, possibly we could save more

    val_loss_rounded = round(val_loss, 4)
    filename = "loss-"+str(val_loss_rounded).replace('.', '_')
    print(f"Best train validation loss: {val_loss}")
    with open(f"{model_dir}/{filename}.txt", "w") as file:
        file.write(f"Validation Loss: {val_loss}\n")
        print(f"Validation loss saved to {model_dir}/{filename}.txt")




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


