# RealB Dataset

## 1. Evaluation

### 1.1 Download Our Dataset

We apologize for the inconvenience. Due to the YouTube's license issue, to obtain the dataset, please email hqi@berkeley.edu (with the title "RPIN RealB Dataset Download"), stating:

- Your name, title and affilation
- Your intended use of the data
- The following statement:
    > With this email we declare that we will use the RealB Dataset for non-commercial research purposes only. We will not redistribute the data in any form except in academic publications where necessary to present examples.

We will promptly reply with the download link. Then you can use the file_id we replied in the following command:
```
gdown --id ${F_ID_REPLIED} -O data/realb.zip
unzip data/realb.zip -d data/
```

### 1.2 Evaluate Our Prediction Model

You can download our pre-trained RPIN model using the following script:
```
gdown --id 1w8Id8UYfQcYhc3nh2_I6Qnkt56Qr29xS -O outputs/phys/realb/rpcin.zip
unzip outputs/phys/realb/rpcin.zip -d outputs/phys/realb/
```

## 2. Training

You can train your prediction model by:

```
python train.py --cfg configs/realb/rpcin.yaml --gpus ${GPU_ID} --output ${OUTPUT_NAME}
```
