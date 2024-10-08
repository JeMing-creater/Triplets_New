# Experiment environment of triplet datasets

## setting
#### all params of this experiment environment are included in \config.yml, user can change connection params for your detail environment, especially the `config.dataset.data_dir`.

#### this code only require the version of accelerate <= 0.18.0.

## get project
```
git clone https://github.com/JeMing-creater/Triplets_New.git
```

## training
### two types of training:

#### single GPU training: 
```
python3 main.py
```

#### mutliple GPUs training:

```
sh run.sh
```
Ps: the detail in this file is :
```
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
torchrun \
  --nproc_per_node 1 \
  --master_port 29550 \
  main.py
```
user can change ```--nproc_per_node``` to needed numbers of GPUS, while the ```export CUDA_VISIBLE_DEVICES``` is needed to record the index of GPUs（from 0 ==> max）

#### Dataset tasks
user can change tasks for different dataset tasks by changing the param ```trainer.dataset``` in ```config,yml```. Especially, this choose can just set ```T45``` and ```T50```.

#### tips: user can change setting of mutliple GPUs in run.sh. For now, mutliple GPUs training utilizes 2 GPUs as training device, the first two devices are used by default.

## tensorboard
```
tensorboard --logdir=/logs
```
Ps: some environments may not accept relative path, user can change ```/logs``` ==> ```detail path root```.
## huggingface 
If you encounter pre-trained model parameter connection network errors in China, please change the mirror proxy: 
### Linux：
```
pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
```
