## Intro
This repository is for CMPT732 project, which is created with reference to [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement).

## Dataset
[FFHQ](https://github.com/NVlabs/ffhq-dataset) is used for training and [CelebA-HQ](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256) is used for validation.

Download the dataset and prepare it in **LMDB** or **PNG** format using script. (LMDB is the default)
```
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 -l
```
Create a folder `dataset` to the root directory and put your own images inside, for example:
`dataset/ffhq_16_128`

## Requirements
```
pip install -r requirement.txt
```

## Training
16--->128
```
python "sr.py" -p train -c "config/sr_sr3_16_128.json" -enable_wandb
```
64--->256
```
python "sr.py" -p train -c "config/sr_sr3_64_256.json" -enable_wandb
```

## Test/Evaluation
```
python sr.py -p val -c config/sr_sr3.json
python eval.py -p [result root]
```
