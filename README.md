
# CRC-RL: A Novel Visual Feature Representation Architecture for Unsupervised Reinforcement Learning

Improving the sample efficiency of pixel-based model-free RL algorithms
by learning a high-level latent representation of given input observation.
This is an attempt to combine reconstruction loss and consistency loss with contrastive learning for learning efficient representations.
All the experiments are carried out on [DeepMind Control Suite](https://www.deepmind.com/open-source/deepmind-control-suite) environments.




## Installation

All the dependencies are in the setup/env.yml file. We assume that you have access to a GPU with CUDA >=9.2 support. The dependencies can be installed with the following commands:


```bash
conda env create -f setup/env.yml
conda activate crc
sh setup/install_envs.sh
```
    
## Instructions

To run the code, set the required hyperparameters and environment name in the **config.py** file and execute the command-

```bash
python train.py
```
Note- Deep Mind Control Suite requires mujoco as its prerequisite. Please refer to this [link](https://www.youtube.com/watch?v=Wnb_fiStFb8) for the mujoco installation.

[Weights & Biases](https://wandb.ai/site) is used for logging the training and evaluation plots. For initiating wandb logging set the WB_LOG flag to True in the train.py file and login to your wandb account.



## References
- We have used [CURL](https://github.com/MishaLaskin/curl) architecture as our baseline. All our contributions/modifications are done on top of it.
- Our implementation of SAC is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae) by Denis Yarats.
- For testing the generalization capability of our proposed method we have used the **color hard** and **video easy** environments of [dm-control-generalization-benchmark](https://github.com/nicklashansen/dmcontrol-generalization-benchmark)



