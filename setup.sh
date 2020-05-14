#!/bin/bash

#pip install kaggle
mkdir -p ~/.kaggle
cp ./kaggle/kaggle.json ~/.kaggle
kaggle competition download alaska2-image-steganalysis
