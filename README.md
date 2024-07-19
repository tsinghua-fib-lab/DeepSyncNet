# DeepSyncNet

A python  implementation of our manuscript "Deep synergetic modeling of slow-fast dynamics in complex systems”. Thank you for reviewing our manuscript.

DeepSyncNet is a physics-inspired deep learning model to discern slow and fast variables and capture their dynamics from the evolution trajectories of systems. It learns representations of the system state from multi-timescale perspectives through the powerful pre-training and fine-tuning paradigm, and ensures the interpretation of learned slow and fast variables by imposing built-in physical constraints through a novel physics-inspired architecture.  

This repository contains the source code for generating simulated data and reproducing experimental results. The source code is implemented in Python.

## Installation Guide

- Install Anaconda https://docs.anaconda.com/anaconda/install/index.html
- Create an experimental environment with request packages: (this will take about 0.5 hours)

```bash
conda create --name <env> --file requirements.txt
```

**Main Requirements:** (see requirement.txt for details)

```shell
# platform: linux-64 (Ubuntu 11.4.0)

numpy==1.22.4
python==3.10.12
scikit-dimension==0.3.3
scikit-learn==1.3.0
scipy==1.11.2
sdeint==0.3.0
torch==2.0.1+cu118
torchdiffeq==0.2.3
```

## Demo

We have written the code used to reproduce the experimental results of each model under different dynamics in separate scripts. You can modify the parameter settings and run the scripts (set to the parameter settings recommended in the appendix of the article to reproduce the experimental results).

choose the {system}: `FHN`, `FHNv`, `HalfMoon`, `Lorenz`

```shell
# ours
./scripts/{system}/OURS.sh

# baseline
./scripts/{system}/LED.sh
./scripts/{system}/NeuralODE.sh
./scripts/{system}/DeepKoopman.sh
./scripts/{system}/SINDy.sh
./scripts/{system}/RC.sh
```

We recommend turning on the **--parallel** option to enable parallel execution of programs with different random seeds to improve test efficiency. Please be careful to choose the suitable number of random seeds  **--seed_num** according to your computational and cache resources. The result of the experiment should be an average of multiple random seeds. The entire training and testing will take a few minutes to a few hours (depending on your computing resources).

**We have encapsulated all the tedious operations. When you run any experimental script, the main program `run.py` will automatically execute the following steps in sequence:**

1. Simulate and process the dataset
2. Train the model
3. Test the model

## File Description

The description of the project folder is as follows:

```shell
.
├── Data
│   ├── IC # initial conditions for FHNv system
│   ├── dataset.py # pytorch-style dataset class
│   ├── generator.py # data simulator
│   └── system.py # system equations
├── README.md
├── main.py # main functions
├── methods
│   ├── __init__.py
│   ├── baseline.py # train and test functions for other baselines
│   ├── ours.py # train and test functions for our DeepSyncNet
│   ├── pysindy # package for SINDy
│   ├── rc.py # train and test functions for RC
│   └── sindy.py # train and test functions for SINDy
├── models
│   ├── __init__.py
│   ├── ami.py # AMI and Decomposition Network of our DeepSyncNet
│   ├── deepkoopman.py # baseline
│   ├── layers.py # basic layers
│   ├── led.py # baseline
│   ├── neural_ode.py # baseline
│   ├── sfs_ode.py # Sync Network of our DeepSyncNet
│   └── weight_init.py # initial functions for all models
├── requirements.txt # Environmental requirements
├── run.py # main process
├── scripts # automation scripts for all experiments
│   ├── FHN
│   ├── FHNv
│   ├── HalfMoon
│   └── Lorenz
└── util # functions for visualization and ID estimation
    ├── __init__.py
    ├── calibri.ttf
    ├── common.py
    ├── intrinsic_dimension.py
    ├── results_visual.py
    └── visual_config.py
```

## License

This repo is covered under the **MIT License**.
