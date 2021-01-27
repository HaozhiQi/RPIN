# PHYRE Dataset

In this document, we provide a step-by-step instruction on:
- How to evaluate our pretrained prediction model and the corresponding planning model for the PHYRE dataset.
- How to train those models yourself.

Throughout this file, we will use **within-task generalization setting and fold 0** as an example. Other models (both for within-task generalization and cross-task generalization) are in the same format, they can be downloaded and evaluated by simply chage the model name in the following scripts.

## 1. Evaluation

### 1.1 Download Our Dataset

You can download our prepared PHYRE dataset using the following script:
```
# md5sum: 9a9b4f7e484613b7a9091819c801d2e9
# the zip file is ~15G and the whole dataset is ~100G
sh scripts/download.sh 1gmnjF0R59yiaYLsBmLuLs6hYN_UEARpU data/PHYRE_1fps_p100n400.zip
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

### 1.2 Evaluate Our Prediction Model

**Note that this model is trained with PRED_SIZE_TRAIN = 10. For models trained with PRED_SIZE_TRAIN = 5 and PRED_SIZE_TEST = 10, please refer to our [MODEL_ZOO](docs/MODEL_ZOO.md).**

You can download our pre-trained RPIN model using the following script:
```
sh scripts/download.sh 1h-zEsOM0FyPog1Urh5slnpKuKXxG16bd outputs/phys/PHYRE_1fps_p100n400/W0_rpcin.zip
unzip outputs/phys/PHYRE_1fps_p100n400/W0_rpcin.zip -d outputs/phys/PHYRE_1fps_p100n400/
```

Run the following for evaluation:
```
sh scripts/test_pred.sh PHYRE_1fps_p100n400 rpcin W0_rpcin ${GPU_ID}
```
The L2 error should be 3.526 (x 1e-3).

---

### 1.3 Evaluate Our Planning Model

To evaluate the planning model, you need to download our pre-trained trajectory classifier:
```
sh scripts/download.sh 1kRiuzEHU2t4K2W_rp2jo5KZr6Lyhq78B outputs/cls/PHYRE_1fps_p100n400/W0_res18.zip
unzip outputs/cls/PHYRE_1fps_p100n400/W0_res18.zip -d outputs/cls/PHYRE_1fps_p100n400/
```

Run the following command for evaluation:
```
sh scripts/test_plan.sh PHYRE_1fps_p100n400 rpcin W0_rpcin W0_res18 ${GPU_ID} plan
```

This is usually a very slow process (more than 24 hours in my machine). A large part of the reason is that it takes time to get the initial bounding boxes from the simulator. Therefore, we recommend to download our cache for the intial bounding boxes:
```
sh scripts/download.sh 1KTvLa7WSfyd4Y7d-WUKmTgNde67HAcsU ./cache.zip
unzip cache.zip -d ./
```
This will greatly improve the performance. However, if you are still not satisified, you can split the 25 tasks into different groups. And then evaluate them using different python process.

---

## 2.Training

### 2.1 Train Your Prediction Model

You can train your prediction model by:
```
# Feel free to change rpcin_within_t10 to other yaml in that folder
python train.py --cfg configs/phyre/pred/rpcin_within_t10.yaml --gpus 0,1 --output ${OUTPUT_NAME}
```

---

### 2.2 Train Your Classification Model

This is a bit complicated. Firstly, you need to generate the training / testing set for the classification model.
Each of the data is a video sequence from a template under different chosen action. You can do this by running:
```
sh scripts/test_plan.sh PHYRE_1fps_p100n400 rpcin ${YOUR_PRED_MODEL} None 0 proposal
```
This usually takes a long time. Similar as the scene cache above, you can download the cache for inital bounding box locations for each scene. But note that, the cache is correct only when you run the exactly same script as we provided. Otherwise the sampled action order may be different.

After doing that, now you can train your classification model by running this script:
```
python train_cls.py --cfg configs/phyre/cls/res18.yaml --gpus 0 --output {OUTPUT_NAME}
```

---

## Customization 

### 1. Dataset Generation

See code and comments in `tools/gen_phyre.py`.

---

**Misc**

We use phyre version 0.2.1 for our paper and this repository. The fold division may change in future versions as mentioned in [this issue](https://github.com/facebookresearch/phyre/issues/40).
