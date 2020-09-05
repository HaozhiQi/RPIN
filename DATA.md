# Dataset Instructions

## Use Pre-built Dataset

### Simulated Billiard

Download the [train](https://drive.google.com/file/d/1WqWdboM4kMjYaq2DRHQqHpbKhuQtDrLx/view?usp=sharing) / [test](https://drive.google.com/file/d/1mt0xAMtCTsW9NCh6LNsy7H64xj5ZTlhc/view?usp=sharing) / [planning](https://drive.google.com/file/d/1I0lNB3fcyAgLwRdJ5bBOMPb3KuX3m4RT/view?usp=sharing) pickle files.

And put them to ```data/simb/``` so that they look like:
```
data/simb/train.pkl
data/simb/test.pkl
data/simb/planning.pkl
```

Then process the files using
```
python tools/prepare_billiard.py --split train
python tools/prepare_billiard.py --split test
python tools/prepare_billiard.py --split planning
```

### Real-World Billiard

Due to the license issue, to obtain the dataset, please email hqi@berkeley.edu.

### PHYRE (with-task generalization)

Download from this [link](https://drive.google.com/file/d/1B7qQkbEHg8SRT6IQYs8Si6qrn1dPWmza/view?usp=sharing).

Then unzip it to ```data/phyre``` so that it looks like:
```
data/phyre/train/
data/phyre/test/
```

### PHYRE-C (cross-task generalization)

Download from this [link](https://drive.google.com/file/d/1NS0diBRUYOrlCg4-IIMusD4oJbQOx1R3/view?usp=sharing).

Then unzip it to ```data/phyrec``` so that it looks like:
```
data/phyrec/train/
data/phyrec/test/
```

### ShapeStacks

Download from this [link](https://drive.google.com/file/d/1A1uhCTr62C2qay7YU_fmmwBYrK3kAwm1/view?usp=sharing).

Then unzip it to ```data/shape-stack``` so that it looks like:
```
data/shape-stack/train/
data/shape-stack/test/
```

## Generate dataset from scratch

The script of generating those datasets are available at ```tools/```:

```
python tools/gen_billiard.py # script used for generating simb
python tools/gen_phyre.py # script used for generating phyre
python tools/gen_shapestack.py # see below
```
To generate shapestack, you need to download the original shapestack data from [CVP](https://github.com/JudyYe/CVP), and place it in ```./data``` folder.
