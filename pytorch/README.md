# PyTorch

- Training of neural networks solutions based on PyTorch
- version 0.1.0

## Table of Contents

- [PyTorch](#pytorch)
  - [Table of Contents](#table-of-contents)
  - [How do I get set up](#how-do-i-get-set-up)
  - [Features](#features)
    - [Detectron 2](#detectron-2)

## How do I get set up

- Install Anaconda environment with:
  > `conda env create -f env_pytorch.yml`

  or if already existing update it with:
  > `conda env update --prefix ./env --file env_pytorch.yml  --prune`

  > NOTE: There may be some errors while installing packages as detectron2 hasn't been installed yet.

- To run, set python interpreter to be the one located in Anaconda virtual environment
  > for VS Code: `F1 > Python: Select Interpreter > Python *version* ('pytorch':conda)`

- Run:
  > `conda activate pytorch`

- Install detectron2:
  > `pip install 'git+https://github.com/facebookresearch/detectron2.git'`
  
  If encountering problems refer to <https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md>

## Features

### Detectron 2

- Detectron2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms.
- <https://github.com/facebookresearch/detectron2>
