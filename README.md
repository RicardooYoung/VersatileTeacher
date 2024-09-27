## Introduction

This is the official repository for the paper [Versatile Teacher: A Class-aware Teacher-student Framework for Cross-domain Adaptation](https://arxiv.org/abs/2405.11754).

## Preparation

### Requirements

This repository is built upon [YOLOv5](https://github.com/ultralytics/yolov5) and follow the same requirements. Please refer to the original repository for more details.

### Datasets

Prepare your own datasets in YOLO format and create a corresponding ``.yaml`` file in the folder ``data``.

## Train

Firstly, use the script
```angular2html
python train.py \
    --cfg models/yolov5l_da.yaml \
    --weights weights/yolov5l.pt \
    --data data/src_data.yaml \
    --data data/tgt_data.yaml \
    --device 0 \
    --epochs 50 \
    --batch-size 16 \
```
to initialize the teacher model, saved as ``weights/yolov5l_teacher.pt``.

Then, use the script
```angular2html
python train.py \
    --cfg models/yolov5l_da.yaml \
    --cfg models/yolov5l.yaml \
    --weights weights/yolov5l_teacher.pt \
    --weights weights/yolov5l_teacher.pt \
    --data data/src_data.yaml \
    --data data/tgt_data.yaml \
    --device 0 \
    --epochs 50 \
    --batch-size 16 \
```
to train the student model.

## Citation
```
@article{YANG2025111024,
title = {Versatile Teacher: A class-aware teacher–student framework for cross-domain adaptation},
journal = {Pattern Recognition},
volume = {158},
pages = {111024},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.111024},
url = {https://www.sciencedirect.com/science/article/pii/S0031320324007751},
author = {Runou Yang and Tian Tian and Jinwen Tian},
}
```
