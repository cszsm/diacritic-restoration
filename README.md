# Diacritic restoration using deep neural networks

Currently only the baseline LSTM network works. To use this first we need to preprocess the data for training. Secondly the network has to be trained. After that we can use the network for accentizing. There are scripts for all the three steps. There are examples below to try the accentizing, the scripts are in the root directory of the repository.

To preprocess the data, run the preprocess.py python script. A mandatory argument for this is the count of the required samples for each vowel. An optional argument is the window_size. If this is not given, the script will do the preprocessing for different window sizes given by an exhaustive search by the network.
```python
python .\preprocess.py 10000 --window_size 1
```

To train the data, run the train.py script. There are two optional arguments, one is for the LSTM units in the network and the window size. If none of them is given, the script will train different networks by parameters provided by the exhaustive search.
```python
python .\train.py --units 16 --window_size 1
```

To accentize with a trained network, run the accentize.py script. There are two mandatory parameters to identify the network: the LSTM units and window size. There is a third mandatory parameter: the text to accentize.
```python
python .\accentize.py 16 1 ekezetesites
```