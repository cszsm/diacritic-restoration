# Diacritic restoration using deep neural networks

Accentize Hungarian text:
```
echo ekezetesites | python .\accentize.py
```
This uses a model with three bidirectional LSTM layers stacked, each layer with 128 LSTM units trained on the Hungarian Webcorpus (http://mokk.bme.hu/resources/webcorpus/)