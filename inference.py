import os
import time
import shutil
import argparse

from lib import model_D, model_G, model_E

dir_path = os.path.dirname(os.path.realpath(__file__))
SHAPENET_VOX_PATH = os.path.join(dir_path, "data")

from data_loaders import data_loader
from data_loaders.util import batch
from utils import *

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from config import Config

parser = argparse.ArgumentParser('')
parser.add_argument('--data', type=str, default='multicate')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--setting', type=str, default='val')
args = parser.parse_args()

cfg = Config()

data_setting = cfg.category_settings[args.data]
data_train = data_setting['train']
data_val   = data_setting[args.setting]

tf.reset_default_graph()
handle_ph = tf.placeholder(tf.string, shape=[])
val_iters = {} # cat -> handles
val_ds    = {} # cat -> dataset
val_size  = {} # cat -> size
for cate, split in data_val:
    p = os.path.join(dir_path, "data",
        "%s_32x32alpha_perspective_d100_r32_vp24_random_default"%cate, split)
    vs = data_loader.StreamingValDataLoader([p],
        shape=[cfg.resolution, cfg.resolution, 1],
        viewpoints=cfg.viewpoints,
        binarize_data=cfg.binarize_data)

    cate_split = "%s-%s"%(cate, split)
    val_ds[cate_split] = batch(vs.dataset, cfg.batch_size)
    val_iters[cate_split] = val_ds[cate_split].make_one_shot_iterator()
    val_size[cate_split] = len(vs)

iterator = tf.data.Iterator.from_string_handle(handle_ph,
        list(val_ds.values()).pop().output_types,
        list(val_ds.values()).pop().output_shapes)
val_inputs = iterator.get_next()

g_net = model_G.Generator(cfg)
e_net = model_E.Encoder(cfg)

# Input variables
is_training = tf.placeholder(tf.bool, name='is_training')

# Validation pass
print(val_inputs)
x_val_1 = val_inputs["image"]
y_val_1 = val_inputs["pose"]
val_vox = val_inputs["vox"]
val_pass_noise, val_pass_pose_logits = e_net(x_val_1, is_training)
val_pass_z = tf.concat([val_pass_noise, y_val_1], axis=1)
_, val_pass_vox = g_net(val_pass_z, is_training)

max_iou_tensor = maxIoU_tf(val_vox, val_pass_vox, step=1e-1)
t04_iou_tensor = tf.reduce_mean(iou_t_tf(val_vox, val_pass_vox, threshold=0.4))
t05_iou_tensor = tf.reduce_mean(iou_t_tf(val_vox, val_pass_vox, threshold=0.5))

saver = tf.train.Saver()

output_file = "%s.%s.txt"%(args.resume, args.setting)
with open(output_file, "w") as outf:
    outf.write("category_split\tMaxIoU\tAP\tIoU(t>0.4)\tIou(t>0.5)\n")
    with tf.Session() as sess:
        # Resume variables
        saver.restore(sess, args.resume)
        for cate, split in data_val:
            print("="*80)
            print("Validating category %s-%s"%(cate, split))
            cate_split = "%s-%s"%(cate, split)
            string_handle = sess.run(val_iters[cate_split].string_handle())

            all_metrics = []
            try:
                with tqdm(total=(val_size[cate_split]//cfg.batch_size+1)) as pbar:
                    cnt = 0
                    while True:
                        gtrs, pred, max_iou, iou_t04, iou_t05 = sess.run(
                            [val_vox, val_pass_vox, max_iou_tensor,
                            t04_iou_tensor, t05_iou_tensor], feed_dict={
                                is_training:False,
                                handle_ph:string_handle
                            }
                        )

                        ap = average_precision(gtrs, pred)
                        all_metrics.append(np.array(
                            [max_iou, ap, iou_t04, iou_t05])[np.newaxis,:])

                        cnt += 1
                        pbar.update(cnt)

            except tf.errors.OutOfRangeError:
                print("End of validation dataset: %s"%cate_split)

            all_metrics = np.concatenate(all_metrics, axis=0).mean(axis=0)
            max_iou, ap, iou_t04, iou_t05 = all_metrics[:]
            print("max_iou:%.3f\tap:%.3f\tiou_t04:%.3f\tiou_t05:%.3f"\
                 %(max_iou, ap, iou_t04, iou_t05))
            print()
            print("="*80)

            # Write to file
            outf.write("%s\t%.4f\t%.4f\t%.4f\t%.4f\n"\
                      %(cate_split, max_iou, ap, iou_t04, iou_t05))

print(output_file)
