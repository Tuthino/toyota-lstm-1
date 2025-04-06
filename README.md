# toyota-lstm-1
# Usage
## Before Starting 
Tensorflow is available in <= python3.11, so if you are using newer, then you need
to eiter install specific python version on your main pc, or use docker for that  

Once you have the correct python version, do not forget to create the venv, using 
this specific python, for example:
`python3.11 -m venv .venv`



## Data
Downloaded from here: [Singapur monthly](https://data.gov.sg/datasets/d_48cb38d12697d3463c8cadfb22e6c61d/view)
Extracted only 'Cargo' using `cat VesselCalls75GTMonthly.csv| grep Cargo > SingapurCargoMonthly.csv`
Filtered out the tonage using `cut -d, -f1,3 SingapurCargoMonthly.csv > SingapurCargoVesselsMonthly.csv