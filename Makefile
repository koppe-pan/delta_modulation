export SR:=16000
export THRESHOLD:=1000
export SAMPLE_PER_PART:=10000
export MINUS:=0
export PLUS:=1
export L:=10
export LR:=0.3
export EPOCH_NUM:=50

setup_data:
		ls data/wav | xargs -I% python3 src/create_data.py --file %
		ls data/wav | xargs -I% python3 src/create_npy.py --file %

train:
		python3 src/train_rnn.py

wav:
		python3 src/create_data.py --file $(TESTFILE)
		python3 src/create_npy.py --file $(TESTFILE)
		python3 src/create_wav.py --file $(TESTFILE)

clean:
		rm -rf data/binary
		rm -rf data/npy
		rm -rf data/result
