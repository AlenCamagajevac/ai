# AI

- Section dealing with AI and image processing
- version 0.0.0

## Table of Contents

- [AI](#ai)
  - [Table of Contents](#table-of-contents)
  - [How do I get set up](#how-do-i-get-set-up)
  - [Subsections](#subsections)
    - [Data Preparation](#data-preparation)
    - [PyTorch](#pytorch)

## How do I get set up

- Ideally every subsection has it's own python virtual environment
- Install Anaconda environment with:
  > `conda env create -f *environment_name*.yml`

- or if already existing update it with:
  > `conda env update --prefix ./env --file *environment_name*.yml  --prune`

- To run, set python interpreter to be the one located in Anaconda virtual environment
  > for VS Code: `F1 > Python: Select Interpreter > Python *version* ('*environment_name*':conda)`

- Run:
  > `conda activate *environment_name*`

## Subsections

> Aditional description of the script functionalities can be found in README files of every subsection

### Data Preparation

- Subsection dealing with data preparation from Unreal Engine for input into various NNs

### PyTorch

- Subsection dealing with PyTorch for neural network training
