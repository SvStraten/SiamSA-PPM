# SiamSA-PPM

This is the official GitHub repository for SIAMese learning with Statistical Augmentation for Predictive Process Monitoring (SimSA-PPM)

![The Framework](framework.png)

## Datasets

All the datasets provided in the paper can be downloaded from the following [Google Drive link]() TO BE ADDED. After downloading, locate the folders in the 'datasets' folder.


## Command-Line Arguments

This script accepts several command-line arguments:

```
--dataset| "sepsis" | Name of the dataset in lowercase letters. 
--STRATEGY| "combi" | Ours is named "combi", if you want to use only random augmentation use "random".

```

## **Commands for Pretraining**
```
python pretraining.py --dataName "sepsis"
```