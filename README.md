# Region Proposal Interaction Network

This repository is an official PyTorch implementation of the ICLR paper:

<b>Learning Long-term Visual Dynamics with Region Proposal Interaction Networks</b> <br>
[Haozhi Qi](https://haozhi.io/),
[Xiaolong Wang](https://xiaolonw.github.io/),
[Deepak Pathak](https://www.cs.cmu.edu/~dpathak/),
[Yi Ma](http://people.eecs.berkeley.edu/~yima/),
[Jitendra Malik](https://people.eecs.berkeley.edu/~malik/) <br>
International Conference on Learning Representations (ICLR), 2021 <br>
[[Website](https://haozhiqi.github.io/RPIN)], [[Paper](http://arxiv.org/abs/2008.02265)]


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

## Using this codebase

### Installation

This codebase is developed and tested with python 3.6, PyTorch 1.4, and cuda 10.1. But any version newer than that should work.

Here we gave an example of installing RPIN using the conda virtual environment:
```
conda create -y -n rpin python=3.6
conda activate rpin
# install pytorch according to https://pytorch.org/
conda install -y pytorch==1.4 torchvision cudatoolkit=10.1 -c pytorch
pip install yacs hickle tqdm matplotlib
# OpenCV changes their way of reading image and has different results
# We don't use later version for consistency
pip install opencv-python==3.4.2.17 
# to evaluate phyre planning, you also need to also do
pip install phyre
```

Then
```
git clone https://github.com/HaozhiQi/RPIN
cd RPIN
```

### Data Preparation & Training & Evaluation 

This codebase supports all of the four dataset we used in our paper. We provide the instructions for each dataset separately. Please see [PHYRE](docs/PHYRE.md), [SS](docs/SS.md), [RealB](docs/RealB.md), and [SimB](docs/SimB.md) for detailed instructions.

For results and models we provided in this codebase, see the [Model Zoo](docs/MODEL_ZOO.md).

## Citing RPIN

If you find **RPIN** or this codebase helpful in your research, please consider citing:
```
@InProceedings{qi2021learning,
  author={Qi, Haozhi and Wang, Xiaolong and Pathak, Deepak and Ma, Yi and Malik, Jitendra},
  title={Learning Long-term Visual Dynamics with Region Proposal Interaction Networks},
  booktitle={ICLR},
  year={2021}
}
```
