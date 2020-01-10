# Data Preparation  <!-- omit in toc -->

- Collection of features dealing with preparing raw data from Unreal Engine for input in various NNs
- version 0.1.0

## Table of Contents  <!-- omit in toc -->

- [How do I get set up](#how-do-i-get-set-up)
- [Features](#features)
  - [UE to COCO](#ue-to-coco)

## How do I get set up

- Dependencies alongside python interpreter are located in `.venv` virtual environment
- To run set python from venv in VS Code as the interpreter

## Features

### UE to COCO

- These scripts convert images located in `Screenshots` togehter with data from Unreal Engine located in `Metadata\Segmentation.csv` into a json file compatible with COCO dataset format for NN training
- Output is saved in `Metadata\coco_json.json`