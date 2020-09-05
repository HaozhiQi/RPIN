# Region Proposal Interaction Network

This repository is an official PyTorch implementation of the paper:

<b>Learning Long-term Visual Dynamics with Region Proposal Interaction Networks</b> <br>
[Haozhi Qi](https://people.eecs.berkeley.edu/~hqi/),
[Xiaolong Wang](https://xiaolonw.github.io/),
[Deepak Pathak](https://www.cs.cmu.edu/~dpathak/),
[Yi Ma](http://people.eecs.berkeley.edu/~yima/),
[Jitendra Malik](https://people.eecs.berkeley.edu/~malik/) <br>
[[website](https://haozhiqi.github.io/RPIN)], [[arXiv](http://arxiv.org/abs/2008.02265)]


![image](https://haozhiqi.github.io/RPIN/figs/teaser.png)

## Introduction

In this project, we aim to leverage the ideas from success stories in visual recognition tasks to build object 
representations that can capture inter-object and object-environment interactions over a long range. To this end, 
we propose Region Proposal Interaction Networks (RPIN), which reason about each object's trajectory in a latent 
region-proposal feature space. Our approach outperforms prior methods by a significant margin both in terms of 
prediction quality and their ability to plan for downstream tasks, and also generalize well to novel environments.

## Method

![image](https://haozhiqi.github.io/RPIN/figs/methodv2.png)

Our model takes N video frames as inputs and predicts the object locations for the future T timesteps, as illustrated above. We first extract the image feature representation using a ConvNet for each frame, and then apply RoI pooling to obtain object-centric visual features. These object feature representations are forwarded to the interaction modules to perform interaction reasoning and predict future object locations. The whole pipeline is trained end-to-end by minimizing the loss between predicted and the ground-truth object locations. Since the parameters of each interaction module is shared so we can apply this process recurrently over time to an arbitrary T during testing.

## Using RPIN

### Installation

This codebase is developed and tested with python 3.6, PyTorch 1.4, and cuda 10.1. But any version newer than that should work.

Here we gave an example of installing RPIN using conda virtual environment:
```
git clone https://github.com/HaozhiQi/RPIN
cd RPIN
conda create -y -n rpin python=3.6
conda activate rpin
# install pytorch according to https://pytorch.org/
conda install -y pytorch==1.4 torchvision cudatoolkit=10.1 -c pytorch
pip install yacs
pip install opencv-python==3.4.2.17
# to evaluate phyre planning, you need to also do
pip install phyre
```

### Evaluation & Training of Prediction

We provide instructions for each dataset separately.

1. Download data: see [DATA.md](DATA.md).
2. Download our pre-trained models and logs from this [link](https://drive.google.com/file/d/17ZvnHodfOyag8rdO_cBC2Z1ov64uivPk/view?usp=sharing). And unzip it so that the models are placed at ```outputs/phys/${DATASET}/rpin/*```

To run evaluation on the test dataset, use the following commands:

```
sh test_pred.sh ${DATASET_NAME} ${MODEL_NAME} ${CACHE_NAME} ${GPU_ID}
```

For example:

```
sh test_pred.sh simb rpin rpin ${GPU_ID}
sh test_pred.sh realb rpin rpin ${GPU_ID}
sh test_pred.sh phyre rpin rpin ${GPU_ID}
sh test_pred.sh phyrec rpin rpin ${GPU_ID}
sh test_pred.sh shape-stack rpin_vae rpin ${GPU_ID}
```

The results should be as follows:

|         | SimB   | RealB | PHYRE  | PHYRE-C | ShapeStacks
| :---:   | :---:  | :---: | :---:  | :---:   | :---:
| [0, T]  | 2.443  | 0.341 | 4.456  | 6.873   | 1.552
| [T, 2T] | 22.199 | 2.194 | 12.196 | 15.775  | 6.891

To train your own model on our dataset, use the following command:
```
# Training, change ${DATASET_NAME} to simb / realb / phyre / phyrec / shape-stack
python train.py --cfg configs/${DATASET_NAME}/rpin.yaml --gpus ${GPU_ID} --output ${OUTPUT_NAME}
# or for shape-stack:
python train.py --cfg configs/ss/rpin_vae.yaml --gpus ${GPU_ID} --output ${OUTPUT_NAME}
```

### Evaluation of Planning

To evaluate planning performance on PHYRE and SimB, use:
```
sh test_plan.sh phyre rpin rpin ${GPU_ID}
sh test_plan.sh phyrec rpin rpin ${GPU_ID}
sh test_plan.sh simb rpin rpin ${GPU_ID}
```

The results should be as follows:

|         | PHYRE  | PHYRE-C
| :---:   | :---:  | :---:
| Top-1 Success Rate  | 33.08 | 18.33
| Top-100 Success Rate | 83.46 | 74.67


| SimB Init-End Error  | SimB Hitting Accuracy
| :---:  | :---: |
| 7.578      | 62.20 


## Citing RPIN

If you find **RPIN** or this codebase helpful in your research, please consider citing:
```
@article{qi2020learning,
  author={Qi, Haozhi and Wang, Xiaolong and Pathak, Deepak and Ma, Yi and Malik, Jitendra},
  title={Learning Long-term Visual Dynamics with Region Proposal Interaction Networks},
  journal={arXiv},
  year={2020}
}
```
