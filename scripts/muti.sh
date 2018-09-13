#! /bin/bash

python train.py --data "few_shot_train_pose1.0"
python train.py --data "few_shot_train_pose0.5"
python train.py --data "few_shot_train_pose0.1"
python train.py --data "few_shot_train_pose0.01"
