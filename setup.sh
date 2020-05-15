#!/bin/bash

pip install kaggle
mkdir -p ~/.kaggle
cp ./kaggle/kaggle.json ~/.kaggle
kaggle competitions download alaska2-image-steganalysis
mkdir -p data
mv alaska2-image-steganalysis.zip ./data
cd data
unzip alaska2-image-steganalysis.zip
rm alaska2-image-steganalysis.zip 
python rename.py
