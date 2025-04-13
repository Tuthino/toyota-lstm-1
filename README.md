# toyota-lstm-1
# Usage
## Before Starting 
Tensorflow is available in <= python3.11, so if you are using newer, then you need
to eiter install specific python version on your main pc, or use docker for that  

Once you have the correct python version, do not forget to create the venv, using 
this specific python, for example:
`python3.11 -m venv .venv`

### Installing the requirements
Once you have ten venv activated, install requirements
`pip install -r requirements.txt`


## Data
Downloaded from here: [Singapur monthly](https://data.gov.sg/datasets/d_48cb38d12697d3463c8cadfb22e6c61d/view)
Extracted only 'Cargo' using `cat VesselCalls75GTMonthly.csv| grep Cargo > SingapurCargoMonthly.csv`
Filtered out the tonage using `cut -d, -f1,3 SingapurCargoMonthly.csv > SingapurCargoVesselsMonthly.csv
### Ready to use data
The CSV that I'm using with only Cargo details is in this folder `SingapurCargoVesselsMonthly.csv`, just move it inside the `csv/` folder

## Models
### Saving
For every model, new folder is created.  
Inside the folder, for each fold the best model is being saved.  
The best validation loss and best model name is saved in `loss-<val_loss>.txt` file
Model config, model optimizer config and model summary are saved in json files, so you can see 
all layers.  
Parameters configurable inside the code, are saved in `run_params.json` so you can recreate the model
if you would like to train once again with the exact same parameters.  
Prediction diagram is saved as `.png` it shows predictions from the BEST model of ALL folds.

### Worth noting parameters
`sequence_length` tells how many months to look back when computing the prediction
`holdout_fraction` % of the data that is being used as holdout, the model DOES NOT see them during the training. It is being used in the evaluation of the best of all models, to see how good it works with the data that it has NEVER seen before.

### Evaluation
Right now, it predicts only ONE month ahead, based on `sequence_length` previous months.
For example, if the holdout set is 10 inputs, for every input, it takes `sequence_length` previous months and predicts the next ONE month. This evaluation is being done for every input in holdout set one by one.  