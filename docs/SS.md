# ShapeStacks Dataset

In this document, we provide a step-by-step instruction on:
- How to evaluate our pretrained model on the ShapeStacks Dataset.
- How to train those models yourself.

The ShapeStacks dataset is introduced in [CVP](https://github.com/JudyYe/CVP). We parse the raw data using [this code](tools/gen_shapestack.py).

## 1. Evaluation

### 1.1 Download Our Dataset

You can download our prepared ShapeStacks dataset using the following script:
```
gdown --id 1FYtwY03U_xg5lU8j1NHZdQXFDMH8Fjsy -O data/ss.zip
unzip data/ss.zip -d data/
```
The data structure should look like:
```
data/ss/train  # The ShapeStacks Training set
data/ss/test   # The ShapeStacks Testing set, containing videos wiht 3 stacked blocks
data/ss/ss4    # The ShapeStacks Testing set, containing videos with 4 stacked blocks
```

---

### 1.2 Evaluate Our Prediction Model

You can download our pre-trained RPIN model using the following script:
```
gdown --id 1VufPAnn2uSeAe1I9KA-NctpvGTjuLscX -O outputs/phys/ss/rpcin.zip
unzip outputs/phys/ss/rpcin.zip -d outputs/phys/ss/
```
Run the following for evaluation:
```
sh scripts/test_pred.sh ss rpcin_vae rpcin ${GPU_ID}
```

## 2. Training

You can train your prediction model by:
```
python train.py --cfg configs/ss/rpcin.yaml --gpus ${GPU_ID} --output ${OUTPUT_NAME}
```

