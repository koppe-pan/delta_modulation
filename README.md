# Setup before train
1. place every original wav file into `./data/wav/`.
1. execute `make setup_data` command.

ex.)
```
mv ~/Downloads/*.wav ./data/wav/
make setup_data
```

# Create wav after train
1. place the original wav file into `./data/wav/`.
1. execute `make wav` command along with the argument `TESTFILE`

ex.)
```
mv ~/Downloads/hoge.wav ./data/wav/hoge.wav
make wav TESTFILE=hoge.wav
```
