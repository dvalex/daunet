# DAUNet
Deep Augmented Neural Network for Pavement Crack Segmentation

## INSTALLATION

```
git clone https://github.com/dvalex/daunet
cd daunet/python
pip install -r requirements.txt
```
## DOWNLOAD DATASET & DATA PREPARATION

Unix users can use data/download.sh script to automate:

```
cd daunet/data
bash download.sh
```
### Manual

For training: download crack500.zip from [Google Drive](https://drive.google.com/file/d/1q6pQb0xifQULmvIHh9qXlNT6-qz0k5lx/view?usp=sharing)
Unzip it into data/cracks500 subfolder

For evaluating: download testcrop.zip from [Google Drive](https://drive.google.com/file/d/1u7wuaQHWWUtF5ON0MhGXcjwbfItniIK5/view?usp=sharing)
Unzip it into data/testcrop subfolder

## TRAINING
```
cd daunet/python
export SM_FRAMEWORK=tf.keras
```

### First stage
```
python train.py
```

### Second stage
```
python finetune.py
```

## INFERENCE AND EVALUATION
To run inference at all images in directory (by default data/testcrop) run 
```
cd daunet/python
python inference.py
```

After that one can calculate AIU, ODS, OIS, sODS, sOIS using matlab evaluation scripts

 