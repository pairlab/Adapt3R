# Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning


Albert Wilcox, Mohamed Ghanem, Masoud Moghani, Pierre Barroso, Benjamin Joffe, Animesh Garg

[![Static Badge](https://img.shields.io/badge/Project-Page-green?style=for-the-badge)](https://pairlab.github.io/Adapt3R)
[![arXiv](https://img.shields.io/badge/arXiv-2503.04877-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2503.04877)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

This repository contains the author implementation of *Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning*. 

Highlights:
 - Clean implementation of the Adapt3R perception encoder
 - Clean implementations of several SOTA IL algorithms (ACT, Diffusion Policy, BAKU, 3D Diffuser Actor) in a shared framework
 - Code for modifying LIBERO and MimicGen to support novel embodiments and viewpoints.

## Installation

These installation instructions have been tested on a workstation running Ubuntu 20.04

### 1. First, clone the repository. If you don't want to also pull the website code and videos (about 200 MB), do this using

```
git clone --single-branch --branch main https://github.com/pairlab/Adapt3R.git
```

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Follow these steps to set up the development environment:

### 2. Install uv:
If you've never used `uv` before for package management, you'll need to install it.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create and sync a virtual environment:
```bash
uv sync
```

### 4. Install dependencies:

If you want to run point cloud stuff you need to also install DGL according to the instructions [here](https://www.dgl.ai/pages/start.html). On my system I used the following:
```bash
uv pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

### 5. (optional) Install LIBERO
First download it
```bash
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd Adapt3R
```
Next, since LIBERO is old we need to manually add the pyproject.toml
```bash
cp adapt3r/envs/libero/pyproject.toml ../LIBERO/
```
Finally, install it
```bash
uv pip install -e ../LIBERO
```

### 6. (optional) Install MimicGen
First download it
```bash
cd ..
git clone https://github.com/NVlabs/mimicgen.git
cd Adapt3R
```
Next, since MimicGen is old we need to manually add the pyproject.toml
```bash
cp adapt3r/envs/mimicgen/pyproject.toml ../mimicgen/
```
Finally, install it
```bash
uv pip install -e ../mimicgen
```


## Data Download / Processing

### LIBERO

First download the dataset. Note that our code base assumes the data is stored in a folder titled `data/` within the Adapt3r repository. If you would like to store it somewhere else please create a symlink. 

Download the data into `data/` by running
```bash
uv run scripts/download_libero.py
```
Note that this file renames the data such that, for LIBERO-90, it is stored in `data/libero/libero_90_unprocessed`

Then, process it into a format that is suitable for our training by running 
```bash
uv run scripts/process_libero_data.py  task=libero
```
You can minimally modify these instructions to support whichever other LIBERO benchmark you'd like to evaluate on.

### MimicGen

Download the MimicGen data based on the instructions [here](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html). Make sure it ends up in the folder `data/mimicgen/core`.

Next, you'll need to process it to add absolute actions, depth and calibration information. To do this, run the following command (updating the task name to correspond to whichever task you are interested in processing):
```bash
uv run scripts/process_mimicgen.py --hdf5_path data/mimicgen/core/task_dx.hdf5 --output_dir data/mimicgen/core_depth/task_dx.hdf5 --depth
```
Note: This will take a few hours :(.

## Training

Example script to train a diffusion policy with a ResNet backbone on the LIBERO-90 benchmark

```bash
uv run scripts/train.py \
    --config-name=train.yaml \
    task=libero \
    algo=diffusion_policy \
    algo/encoder=rgb  \
    algo.chunk_size=8
```

To train another policy, replace `algo` with the desired policy (options are `act` `baku`, `diffusion_policy` and `diffuser_actor`). To train with another encoder, replace `algo/encoder` with one of `rgb`, `rgbd`, `dp3` or `adapt3r`.

`exp_name` and `variant_name` are used to organize your training runs. In particular, they determind where they are saved. Generally in our workflow, `exp_name` would refer to an experiment encompassing several runs while `variant_name` would be one configuration of parameters encompassing several seeds. For example, if you were sweeping over chunk sizes you might choose `exp_name=chunk_size_sweep` and launch several runs with `chunk_size=${chunk_size} variant_name=chunk_size_${chunk_size}`. Then in WandB, filter by `exp_name` and group by `variant_name`

For debugging, we recommend replacing `--config-name=train.yaml` with `--config-name=train_debug.yaml`, which will change several parameters to make debugging more convenient (enabling tqdm, disabling WandB logging, etc.). 

To train a MimicGen policy, set `task=mimicgen` and `task.task_name=[name]`. For example, to train BAKU with Adapt3R backbone on Square D1, (assuming you've already processed the `square_d1.hdf5` file) run
```bash
uv run scripts/train.py \
    --config-name=train.yaml \
    task=mimicgen \
    task.task_name=square_d1 \
    algo=baku \
    algo/encoder=adapt3r  \
    algo.chunk_size=8
```

Other useful training tips:
 - Our pipeline automatically saves a checkpoint after each epoch that is overwritten after the subsequent epoch. If training crashes, you can resume training by simply rerunning the original command (assuming you are not using the debug mode, which creates a unique directory for each run).
 - You may want to adjust the dataloader configuration to better fit your system.
 - Our pipeline, by default, saves a checkpoint which is not overwritten every 10 epochs. You can change this behavior in the configs


## Evaluation

All of the parameters to build policies are stored in the saved checkpoints so you don't need to specify them at eval time. All you need to do is specify the task and point to the checkpoint path. Example to evaluate a mimicgen coffee_d1 policy

```bash
uv run scripts/evaluate.py \
    task=mimicgen \
    task.task_name=coffee_d1 \
    checkpoint_path=[path]
```
to export videos instead, replace `evaluate.py` with `export_videos.py`.

To run evaluation with an unseen embodiment, add `task.robot=ROBOT`, where `ROBOT` can be one of {`UR5e`, `Kinova3`, `IIWA`}. To run with an unseen viewpoint, add `task.cam_shift=SIZE` where `SIZE` is one of {`small`, `medium`, `large`} or an angle in radians. Our code base also has old infrastructure for adding distractor objects and randomizing the lighting and colors of objects in the scene, but that is untested and unsupported.


Note: you can override parameters saved in the model (maybe temporal aggregation) using the following
```bash
uv run scripts/evaluate.py \
    task=mimicgen \
    task.task_name=coffee_d1 \
    +overrides.temporal_agg=false \
    checkpoint_path=[path]
```

You can change the robot by doing `task.robot=X` and move the camera by doing `task.cam_shift=theta`.

## Design Notes

In this section I ramble about design decisions I've made to help make the repository more readable. Feel free to add if there are things that I've missed that would be helpful to understand

### Hydra Use

I am generally pretty religious with my use of hydra.utils.instantiate. This gives the nice property that most of the objects in the code base are recreatable purely based on python dictionaries, which leads to some nice properties. For example:
 - All parameters passed into any object are visible in the WandB logs. 
 - For evaluation, we can just point to the checkpoint path (where the parameter dictionary is saved) and evaluate without needing to pass in any other information about the policy.

I think that with an understanding of Hydra this makes it much nicer to work with this codebase. However, this paradigm is different from standard Python practices, so I'd suggest giving the [Hydra docs](https://hydra.cc/docs/intro/) a close read before trying to make any substantial changes to this repo.

### Modular policies

Each policy is composed of an observation encoder and an action decoder. The observation encoder processes observations into perception and proprioception tokens and specifies how many of each it will output and what dimension they will be. The action decoder should be designed to condition on this information to condition on arbitrary sequences of encoding tokens from the encoders. It should be the case that any algorithm (ACT, DP, etc) can use any observation encoder.

### Action Representations

The goal is to have maximally flexible action representations without the need to reprocess the dataset for each one. To that end, we do the following with actions:
1. In the shape meta we specify a list of action keys and their corresponding dimensions. Note: in the current version of the repo we assume that for unimanual policies the order is pos, rot, gripper and for bimanual policies the order is r_pos, r_rot, l_pos, l_rot, r_gripper, l_gripper. Note that we mention bimanual policies here because there is some support for it in our codebase but we do not include any bimanual environments.
2. The policy has a preprocess actions function that preprocesses actions from the dataset into the format that the network expects. Note that we run this function before computing normalization statistics. The means that by changing the action preprocessing you can implement arbitrary transformations for the actions and the normalization statistics will be computed correctly.
3. There are two postprocessing functions because certain steps in action postprocessing may need to happen before or after temporal aggregation. For example, transforming absolute actions in the end effector coordinate frame back to the world frame would have to happen before action aggregation so that when we aggregate all the actions we are aggregating over are in the same coordinate frame. However we don't want to aggregate across absolute axis angle rotations due to their discontinuities (prefer 6D rotations) so we need to postprocess again after aggregation to transform to the right action representation. These are examples that are implemented here but hopefully there are other settings where this design decision is useful.

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

