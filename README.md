# DAUNet
Deep Augmented Neural Network for Pavement Crack Segmentation

# INSTALLATION #

git clone https://github.com/dvalex/daunet
cd daunet/python
pip install -r requirements.txt

# DOWNLOAD DATASET#

Download crack500.zip from
https://drive.google.com/file/d/1q6pQb0xifQULmvIHh9qXlNT6-qz0k5lx/view?usp=sharing
and unzip it into data subfolder

# TRAINING #

export SM_FRAMEWORK=tf.keras
python train.py

