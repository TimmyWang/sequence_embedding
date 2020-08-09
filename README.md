## Purpose of the project:
The project makes up some fake sequential data and tries to find the emdeddings for it.

## seq_embedding_exp.py:
<p>contains a class called 'FakeData' and a function called 'get_model'.
<p>FakeData is used to generate fake sequential data based on some rules, including (1) the range for the starting point of the sequential data, (2) the trend, (3) the noise, and (3) the length of the data.
<p>get_model is called to get a autoencoder model for training.

## Sequence_embedding.ipynb:
the jupyter notebook to run the experiment.

## best_model.hdf5:
keras model saved from the current loop of training.

## models folder:
folder to store the models from past.
