# PHYRE Dataset

2021-03-31 Update: I made a major refactor about the PHYRE dataset, and it significantly simplifies the physical reasoning pipeline. However, some corner cases may not be thoroughly tested because of the limited time. If you find any problems, please raise an issue in the github.

In this document, we provide a step-by-step instruction on:
- How to evaluate our pretrained prediction model and the corresponding planning model for the PHYRE dataset.
- How to train those models yourself.

Throughout this file, we will use **within-task generalization setting and fold 0** as an example. Other models (both for within-task generalization and cross-task generalization) are in the same format, they can be downloaded and evaluated by simply changing the model name in the following scripts.

## 1. Evaluate Our Prediction Models

### 1.1 Download Our Dataset

You can download our prepared PHYRE dataset using the following script:
```
# md5sum: 9a9b4f7e484613b7a9091819c801d2e9
# the zip file is ~15G and the whole dataset is ~100G
gdown --id 1gmnjF0R59yiaYLsBmLuLs6hYN_UEARpU -O data/PHYRE_1fps_p100n400.zip
unzip data/PHYRE_1fps_p100n400.zip -d data/
```
Here the name `1fps` means we use stride 60 to generate the phyre data (see [this link](https://github.com/facebookresearch/phyre/blob/920cd2cc2d7ee29c08ae6ebff8f0463c2245d603/src/simulator/task_utils.h#L25)), `p100n400` means when we sample action for a template of a task, we use 100 positive (successful) actions and 400 negative (failure) actions.

If you are unable to run the script above, try to download using the [link](https://drive.google.com/file/d/1gmnjF0R59yiaYLsBmLuLs6hYN_UEARpU/view?usp=sharing):

The data structure should look like:
```
data/PHYRE_1fps_p100n400/images/  # initial images, used as input for RPIN
data/PHYRE_1fps_p100n400/labels/  # boxes and masks labels, for the whole sequence
data/PHYRE_1fps_p100n400/splits/  # the train/test split for different fold
data/PHYRE_1fps_p100n400/full/    # full image sequences for each task
```

---

### 1.2 Evaluate

You can download our pre-trained RPIN model using the following script:
```
gdown --id 10g0U00-pv2dRH2PjfrSi1jlnF4OewrX4 -O outputs/phys/PHYRE_1fps_p100n400/W0_rpcin_t5.zip
unzip outputs/phys/PHYRE_1fps_p100n400/W0_rpcin_t5.zip -d outputs/phys/PHYRE_1fps_p100n400/
```

Run the following for evaluation:
```
sh scripts/test_pred.sh PHYRE_1fps_p100n400 rpcin W0_rpcin_t5 ${GPU_ID}
```
The L2 error should be 1.308 (x 1e-3) for [0, T] and 11.060 (x 1e-3) for [T, 2T] (T=5).

---

### 1.3 Evaluate Our Planning Model

To evaluate the planning model, you need to download our prediction model with the task classifier:
```
gdown --id 1ho6ndZH7BlwNfAyOSqVlS__zYlgpZMhy -O outputs/phys/reasoning/w0.zip
unzip outputs/cls/PHYRE_1fps_p100n400/w0.zip -d outputs/phys/reasoning/
```

Run the following command for evaluation:
```
sh scripts/test_plan.sh reasoning rpcin w0 ${GPU_ID} plan
```

This is usually a very slow process (more than 24 hours in my machine). A large part of the reason is that it takes time to get the initial bounding boxes from the simulator. Therefore, we recommend to download our cache for the intial bounding boxes:
```
gdown --id 1KTvLa7WSfyd4Y7d-WUKmTgNde67HAcsU -O ./cache.zip
unzip cache.zip -d ./
```
This will greatly improve the performance (about 4-5 hours in my machine). However, if you are still not satisified, you can split the 25 tasks into different groups. And then evaluate them using different python process.

---

## 2.Training

### 2.1 Train Your Prediction Model

You can train your prediction model by:
```
# Feel free to change rpcin_within_pred to other yaml in that folder
python train.py --cfg configs/phyre/rpcin_within_pred.yaml --gpus 0,1 --output ${OUTPUT_NAME}
```

---

### 2.2 Train Your Prediction Model for Physical Reasoning

You can train your prediction model by:
```
# Feel free to change rpcin_within_pred to other yaml in that folder
python train.py --cfg configs/phyre/rpcin_within_plan.yaml --gpus 0,1 --output ${OUTPUT_NAME}
```

---

## Customization 

### 1. Dataset Generation

See code and comments in `tools/gen_phyre.py`.

---

**Misc**

We use phyre version 0.2.1 for our paper and this repository. The fold division may change in future versions as mentioned in [this issue](https://github.com/facebookresearch/phyre/issues/40).
