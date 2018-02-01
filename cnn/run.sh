#!/bin/bash

clear=0
file_path="./__model_checkpoints__"

if [ 1 -eq $clear ]; then
	if [ -d $file_path ]; then
		rm -r $file_path
		echo "cleared"
	fi
fi

data_path="./CUP-data"
if [ ! -d $data_path ]; then
	echo "data directory $data_path/ not found"
	echo ""
	echo "needs the following structure:"
	echo "$data_path/"
	echo "├──train/"
	echo "│  ├──images/"
	echo "│  └──xmls/"
	echo "├──eval/"
	echo "│  ├──images/"
	echo "│  └──xmls/"
	echo "└──test/"
	echo "   ├──images/"
	echo "   └──xmls/"
else
	python3 ./src/cnn_cups.py
fi
