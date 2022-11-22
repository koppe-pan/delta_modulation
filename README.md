# How to run our original method.
## Setup before train
1. place every original wav file into `./data/wav/`.
1. execute `make setup_data` command.

ex.)
```
mv ~/Downloads/*.wav ./data/wav/
make setup_data
```

## Train
1. Tune the parameters in `Makefile`.
  - `THRESHOLD` represents the threshold of quantization.
  - `SR` represents the sampling rate.
  - `L` represents the length of the learning vector.
  - `LR` represents the learning rate of the optimizer.
  - `EPOCH_NUM` represents the iteration counts.
1. execute `make train` command.

## Create wav after train
1. place the original wav file into `./data/wav/`.
1. execute `make wav` command along with the argument `TESTFILE`

ex.)
```
mv ~/Downloads/hoge.wav ./data/wav/hoge.wav
make wav TESTFILE=hoge.wav
```
