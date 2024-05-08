## When to Replan? An Adaptive Replanning Strategy for Autonomous Navigation using Deep Reinforcement Learning
[![arxiv](https://img.shields.io/badge/2024-ICRA-red.svg)](https://2024.ieee-icra.org/)

This repository contains the code for the paper "When to Replan? An Adaptive Replanning Strategy for Autonomous Navigation using Deep Reinforcement Learning", ICRA 2024.[[project page](https://omron-sinicx.github.io/when2replan/)] [[paper](https://arxiv.org/abs/2304.12046)]

## Tested environment

- Ubuntu 20.04
- Python 3.8.10


## Installation

```bash
# create venv
cd when2replan
python3 -m venv .venv
source .venv/bin/activate

# pytorch-cpu
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# stable baselines3 bleeding edge version: commit tag https://github.com/DLR-RM/stable-baselines3/commit/0532a5719c2bb46fd96b61a7e03dd8cb180c00fc
pip3 install wheel
pip3 install git+https://github.com/DLR-RM/stable-baselines3@0532a5719c2bb46fd96b61a7e03dd8cb180c00fc

# install other dependencies
cd when2replan
pip3 install -e .
```

## Run Navigation with pre-trained model

Run setting file is [here](./config//run/default.yaml)

```bash
cd scripts
python3 run.py\
        run.model_dir="../trained_model"\
        run.model_name="DijkstraxDWA/16pillar.zip"\
        run.log_save_dir="run/DijkstraxDWA/16pillar"\
        run.navigation_scenarios="config/scenario/sixteen_pillar/random.yaml"\
        run.view_animation=False\
        run.seeds=[37]\
        navigation.common.global_planner="Dijkstra"\
        navigation.common.local_planner="DWA"
```

You can visualize using pygame

```bash
cd scripts
python3 pygame_visualizer.py main ../log/run/latest/[method-name]_data.pkl
```

## Train

Training setting file is [here](./config/train/default.yaml)

```bash
cd scripts
python3 train.py
```

tensorboard

```bash
tensorboard --logdir ./log/tensorboard
```

mlflow
```bash
cd scripts
mlflow ui
```

## Evaluation

Evaluation setting file is [here](./config/eval/default.yaml)

```bash
cd scripts
python3 eval.py
```

You can check the result by mlflow

```bash
cd scripts
mlflow ui
```

Or, logged the all results in `./log` dir



## Citation
```
@misc{honda2024replan,
      title={When to Replan? An Adaptive Replanning Strategy for Autonomous Navigation using Deep Reinforcement Learning},
      author={Kohei Honda and Ryo Yonetani and Mai Nishimura and Tadashi Kozuno},
      year={2024},
      eprint={2304.12046},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
