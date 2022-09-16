The dataset contains the NEURON simulations and python codes for ANN fitting and evaluation.

The first folder contains the ANN benchmarking code, and the necessary ANNs in .h5 file format.

All NEURON models were created with the same logic, the most detailed documentation in in the single compartmental model folder.
Fitting and evaluation was also done based on the same logical principles, the most detailed version is also in the single compartmental folder (fit_CNN_LSTM_latest.py)

The multicompartmental models were constrained by the same file, which can be found in the L2_3 PC folder.

As running these codes are computational resource intensive, there are a few options included to run them.
First, NEURON dataset creation can be parallelized, following the example in L5 PC\init.hoc
Second, as ANN fitting is memory intensive, there is an option for not loa
