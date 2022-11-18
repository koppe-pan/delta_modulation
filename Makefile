setup_data:
		ls data/wav | xargs -I% python3 src/create_data.py --file %
		ls data/wav | xargs -I% python3 src/create_npy.py --file %

wav:
		python3 src/create_data.py --file $(TESTFILE)
		python3 src/create_npy.py --file $(TESTFILE)
		python3 src/create_wav.py --file $(TESTFILE)

clean:
		rm -rf data/binary
		rm -rf data/npy
		rm -rf data/result
