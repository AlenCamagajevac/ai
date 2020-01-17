# Data Preparation

- Collection of features dealing with preparing raw data from Unreal Engine for input in various NNs
- version 0.1.0

## Table of Contents

- [Data Preparation](#data-preparation)
  - [Table of Contents](#table-of-contents)
  - [How do I get set up](#how-do-i-get-set-up)
  - [Features](#features)
    - [UE to COCO dataset](#ue-to-coco-dataset)
    - [UE to U-Net](#ue-to-u-net)

## How do I get set up

- Dependencies alongside python interpreter are located in `.venv` virtual environment
- To run, set python from venv in VS Code as the interpreter

## Features

### UE to COCO dataset

- These scripts convert UE screenshots located in `ue_to_coco\Images` and `ue_to_coco\Masks` togehter with data from Unreal Engine located in `ue_to_coco\Metadata\Segmentation.csv` into a json file compatible with COCO dataset format for NN training
- Output is saved in `ue_to_coco\Metadata\coco.json`

### UE to U-Net

- U-Net requires just binary images (black and white) of the segmented scene that are saved in `ue_to_unet\MasksBinary`
