# Learning Single-View 3D Reconstruction with Limited Pose Supervision
TensorFlow implementation for the paper:

[Learning Single-View 3D Reconstruction with Limited Pose Supervision](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guandao_Yang_A_Unified_Framework_ECCV_2018_paper.pdf)

[Guandao Yang](http://www.guandaoyang.com/), [Yin Cui](http://www.cs.cornell.edu/~ycui/), [Serge Belongie](https://vision.cornell.edu/se3/people/serge-belongie/), [Bharath Hariharan](http://home.bharathh.info/)

## Dependency

+ TensorFlow（>=1.4)
+ OpenCV
+ Matplotlib

The recommended way to install the dependency is
```bash
pip install -r requirements.txt
```

## Preparation

Please use the following Google Drive link to download the datasets: [[drive]](https://drive.google.com/drive/folders/13mokTHTuHOLKnztVv1PKjBpdIEGaUqWv?usp=sharing). There are two files : `data.tar.gz` and `ShapeNetVox32.tar.gz`. Please download both of them and uncompressed it into the project root directory:
```bash
tar -xvf ShapeNetVox32.tar.gz
tar -xvf data.tar.gz
rm ShapeNetVox32.tar.gz
rm data.tar.gz
```

## Training

In order to train a model, please use `train.py` script. The default hyper-parameter are stored in `config.py`, and all the training settings are stored in `training_settings.py`. For example, to train single category chairs with 50% pose annotations, we could use:
```bash
python train.py --data chairs_pose0.5
```
where `chairs_pose0.5` refers to an entry in the `training_settings.py` file.

The `scripts` folder contains command line arguments for running different experiments. Note that each script contains multiple training commands.

For example, in order to run *all* the single category training with 50% of pose annotations, please use:
```bash
./scripts/single_category.sh
```

To train *all* multi-category models or to pre-train *all* models for few-shot transfer learning, please use:
```bash
./scripts/muti.sh
```

## Evaluation

File `inference.py` contains codes to load a trained model and evaluate on specific data split or categories. `scripts/get_score.py` helps to organize the evaluation scores into `.csv` files. For Detail usage, please refer to `scripts/eval_*.sh`.

For example, if you want to evaluate *all* the single category training experiments, try running
```bash
./sripts/eval_single_category.sh
```

## Results

Following results are reported from the testing set. Please compare to these results during reproduction.

#### Single Category Experiments

| Setting           | MaxIoU | AP     | IoU(t=0.4) | Iou(t=0.5) |
|-------------------|--------|--------|------------|------------|
| airplanes_pose0.5 | 0.484  | 0.6377 | 0.4396     | 0.4235     |
| beches_pose0.5    | 0.368  | 0.4822 | 0.3421     | 0.3379     |
| cars_pose0.5      | 0.7421 | 0.8698 | 0.6976     | 0.6777     |
| chairs_pose0.5    | 0.4459 | 0.5703 | 0.4113     | 0.3896     |
| sofas_pose0.5     | 0.5523 | 0.6954 | 0.5276     | 0.5224     |
| tables_pose0.5    | 0.4171 | 0.5541 | 0.355      | 0.3402     |

#### Multiple Category (AP)

| Category  | Pose 100% | Pose 50% | Pose 10% | Pose 1% |
|-----------|-----------|----------|----------|---------|
| airplanes | 0.7103    | 0.7062   | 0.6457   | 0.5316  |
| cars      | 0.9223    | 0.9139   | 0.8888   | 0.797   |
| chairs    | 0.5922    | 0.5738   | 0.5325   | 0.4079  |
| displays  | 0.6025    | 0.5917   | 0.4694   | 0.2857  |
| phones    | 0.8294    | 0.813    | 0.662    | 0.498   |
| speakers  | 0.7035    | 0.6869   | 0.6336   | 0.5481  |
| tables    | 0.5603    | 0.5486   | 0.4827   | 0.3948  |
| Mean      | 0.7029    | 0.6906   | 0.6164   | 0.4947  |

#### Out of category (AP)

| Category | Pose 100% | Pose 50% | Pose 10% | Pose 1% |
|----------|-----------|----------|----------|---------|
| benches  | 0.4243    | 0.4044   | 0.3391   | 0.2485  |
| cabinets | 0.6313    | 0.6123   | 0.5667   | 0.5321  |
| vessels  | 0.6109    | 0.6063   | 0.5581   | 0.5325  |

## TODO
We will release the codes for few-shot experiments soon.

## Citation

If you find this our works helpful for your research, please cite:
```
@InProceedings{Yang_2018_ECCV,
author = {Yang, Guandao and Cui, Yin and Belongie, Serge and Hariharan, Bharath},
title = {Learning Single-View 3D Reconstruction with Limited Pose Supervision},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```
