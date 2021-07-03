Contains a function for sODS/sOIS computation:
SIMPLIFIED_ODS_OIS(RESULTSDIR, DATASET, METHOD)

Parameters:
- ResultsDir - segmentation relults are located
- Dataset - dataset name 
-  Method - method name

Supposed directory strucure:
```
ResultsDir
│
└───Dataset1
│   │
│   └───Method1
│   │   │   img001.mat
│   │   │   img001.png
│   │   │   ...
│   │   
│   └───Method2
│   │    ...
│   │
│   └───gt
│       │   img001.png
│       │   ...
│
└───Dataset2
│   ...
```           
           
Usage example:
```
ResultsDir = '../../RESULTS';
Dataset ='CRACK500';
Method = 'DAUNet';
[sODS, sODS] = simplified_ods_ois(ResultsDir, Dataset, Method)
```
