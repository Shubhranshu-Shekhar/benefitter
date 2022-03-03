*Benefitter: Early Classification on Multivariate Time Series*


# Requirements
For this code to run properly, the following python packages should be installed:
```
numpy
scipy
sklearn
Keras
keras-self-attention [Optional]
```

To run experiments on the UCR dataset, please download the datasets from
http://www.cs.ucr.edu/~eamonn/time_series_data/ and copy it to `data/ucr` while preserving the subfolder structure.


To run the code, use the following command:

python keras_clssification/benefit_predictor.py --data_path "datasets/ucr"  --dataset_name <name> --epochs 1

here <name> = ucr_dataset_name e.g. ECG200.
