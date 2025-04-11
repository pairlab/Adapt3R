# Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning


Albert Wilcox, Mohamed Ghanem, Masoud Moghani, Pierre Barroso, Benjamin Joffe, Animesh Garg

[![Static Badge](https://img.shields.io/badge/Project-Page-green?style=for-the-badge)](https://pairlab.github.io/Adapt3R)
[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2503.04877)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

This repository contains the author implementation of *Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning*. Highlights:
 - Clean implementation of the Adapt3R perception encoder
 - Clean implementations of several SOTA IL algorithms (ACT, Diffusion Policy, BAKU, 3D Diffuser Actor) in a shared framework
 - Code for modifying LIBERO to support novel embodiments and viewpoints.

## Installation

These installation instructions have been tested on a workstation running Ubuntu 20.04

First, clone the repository. If you don't want to also pull the website code and videos (about 200 MB), do this using

```
git clone --single-branch --branch main https://github.com/pairlab/Adapt3R.git
```

Create a Python 3.10 virtual environment using the package manager of your choice. For conda, run

```
conda create -n adapt3r python=3.10 -y
conda activate adapt3r
```
Next, install required packages using
```
pip install -e .
```

You'll need to install DGL according to the instructions [here](https://www.dgl.ai/pages/start.html).

Install LIBERO
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
python -m pip install -e .
```

## Data Download

First download the dataset. Note that our code base assumes the data is stored in a folder titled `data/` within the Adapt3r repository. If you would like to store it somewhere else please create a symlink. 

Download the data into `data/` by running
```
python scripts/download_libero.py
```
Then, process it into a format that is suitable for our training by running 
```
python scripts/process_libero_data.py  task=libero_90_data
```
We also provide support for training on LIBERO 10, although we do not provide results on that benchmark.

## Training

A simple command to train Adapt3r with diffusion policy on LIBERO 90 is:
```
python scripts/train.py \
    --config-name=train.yaml \
    task=libero_90_hybrid \
    algo=diffusion_policy \
    algo/encoder=adapt3r \
    algo.chunk_size=16 \
    exp_name=EXP_NAME \
    variant_name=VARIANT_NAME \
    seed=0
```
To train another policy, replace `algo` with the desired policy (options are `act` and `baku`). To train with DP3 or iDP3, replace `algo/encoder` with `dp3` or `idp3` respectively. To train RGB or RGBD, replace `algo/encoder` with `default` and change `task` to `libero_90_rgb` or `libero_90_rgbd` respectively.

`exp_name` and `variant_name` are used to organize your training runs. In particular, they determind where they are saved. Generally in our workflow, `exp_name` would refer to an experiment encompassing several runs while `variant_name` would be one configuration of parameters encompassing several seeds. For example, if you were sweeping over chunk sizes you might choose `exp_name=chunk_size_sweep` and launch several runs with `chunk_size=${chunk_size} variant_name=chunk_size_${chunk_size}`. Then in WandB, filter by `exp_name` and group by `variant_name`

For debugging, we recommend replacing `--config-name=train.yaml` with `--config-name=train_debug.yaml`, which will change several parameters to make debugging more convenient (enabling TQDM, disabling WandB logging, etc.). 

To run 3D Diffuser Actor, run 
```
python scripts/train.py \
    --config-name=train.yaml \
    task=libero_90_hybrid \
    algo=diffuser_actor \
    algo.chunk_size=8 \
    exp_name=EXP_NAME \
    variant_name=VARIANT_NAME \
    seed=0
```

Other useful training tips:
 - Our pipeline automatically saves a checkpoint after each epoch that is overwritten after the subsequent epoch. If training crashes, you can resume training by simply rerunning the original command (assuming you are not using the debug mode, which creates a unique directory for each run).
 - You may want to adjust the dataloader configuration to better fit your system.
 - Our pipeline, by default, saves a checkpoint which is not overwritten every 10 epochs. You can change this behavior in the configs

## Evaluation

Our model checkpoints store the Hydra configs, allowing them to be loaded using only a path to the checkpoint without passing any additional parameters.

You can run an evaluation using the following command:
```
python scripts/evaluate.py \
    task=TASK \
    algo=ALGO \
    exp_name=EXP_NAME \
    variant_name=VARIANT_NAME \
    checkpoint_path=/path/to/ckpt.pth
    seed=0
```
Be sure to select the correct `task` and `algo` as described above. 

To run evaluation with an unseen embodiment, add `task.robot=ROBOT`, where `ROBOT` can be one of {`UR5e`, `Kinova3`, `IIWA`}. To run with an unseen viewpoint, add `task.camera_pose_variations=SIZE` where `SIZE` is one of {`small`, `medium`, `large`}. Our code base also has old infrastructure for adding distractor objects and randomizing the lighting and colors of objects in the scene, but that is untested and unsupported.

You can replace `evaluate.py` with `export_videos.py` and instead of running the eval loop it will export videos.

## Citation
If you find this work useful, please use the following citation:
```
@misc{wilcox2025adapt3r,
    title={Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning}, 
    author={Albert Wilcox and Mohamed Ghanem and Masoud Moghani and Pierre Barroso and Benjamin Joffe and Animesh Garg},
    year={2025},
    eprint={2503.04877},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2503.04877}}
```


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
