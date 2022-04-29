# SimB Dataset

In this document, we provide instructions on how to prepare data for training/testing our prediction model on SimB Dataset.

## 1. Evaluation

### 1.1 Download Our Dataset

You could download the SimB dataset using the following script:
```
gdown --id 1ucTEMTyzWS1uaknrFU3gTpE3KnNlKAIf -O data/simb.zip
unzip data/simb.zip -d data/
```

If you are unable to run the script above, you can also try to download using the [link](https://drive.google.com/file/d/1ucTEMTyzWS1uaknrFU3gTpE3KnNlKAIf/), and unzip it to have the following file structures:
```
data/simb/train.hkl
data/simb/test.hkl
```

After you have download the hickle files, you need to use the following command to process the data:
```
python tools/prepare_billiard.py --split train
python tools/prepare_billiard.py --split test
```

## 1.2 Evaluate Our Prediction Model

Then, you can evaluate our pre-trained models and train your own models. To evaluate the pretrained models, download them using the following script or the following links for [RPCIN](https://drive.google.com/file/d/1vbJWlLCdT6GqTqry61TB3eEOGtDg9q-J/).

```
gdown --id 1vbJWlLCdT6GqTqry61TB3eEOGtDg9q-J -O outputs/phys/simb/rpcin.zip
unzip outputs/phys/simb/rpcin.zip -d outputs/phys/simb/
```

Then you can run evaluation using:
```
sh scripts/test_pred.sh simb rpcin rpcin ${GPU_ID}
```

## 2. Train the Prediction Model

To train the model, you can simply run the following command:
```
# For RPCIN
python train.py --cfg configs/simb/rpcin.yaml --gpus ${GPU_ID} --output ${OUTPUT_NAME}
```

## 3. Customization

In case you want to generate your customized version of the simulated billiard dataset, read the following file ``tools/gen_billiard.py``. Our SimB dataset is generated using the following command (but note that the dataset may be slightly different because of randomness):

```
python tools/gen_billiard.py # script used for generating simb
```
