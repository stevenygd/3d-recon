import numpy as np
import sklearn.metrics
import tensorflow as tf

def average_precision(gtrs_vox, pred_vox):
    N = gtrs_vox.shape[0]
    assert N == pred_vox.shape[0]
    precisions = []
    for i in range(N):
        gtrs_voxel = gtrs_vox[i,...].flatten()
        pred_voxel = pred_vox[i,...].flatten()
        precisions.append(
            sklearn.metrics.average_precision_score(
            gtrs_voxel, pred_voxel))

    avg_p = np.array(precisions).mean()
    return avg_p

def iou_t(gtrs, pred, threshold=0.5):
    gtrs = np.reshape(gtrs.astype(np.bool), [-1,32*32*32])
    pred = np.reshape(pred > threshold, [-1,32*32*32])
    union = (gtrs | pred).astype(np.int).sum(axis=1)
    inter = (gtrs & pred).astype(np.int).sum(axis=1)
    return inter / union.astype(np.float)

def maxIoU(gtrs_vox, pred_vox, step=1e-1):
    ts = np.arange(0., 1., step)
    ious = []
    for t in ts:
        iou = iou_t(gtrs_vox, pred_vox, threshold=t)
        ious.append(iou.mean())

    ious = np.array(ious)
    return ious.max()

def iou_t_tf(gtrs, pred, threshold=0.5):
    gtrs = tf.cast(tf.reshape(gtrs > threshold, [gtrs.get_shape()[0], 32*32*32]), tf.bool)
    pred = tf.cast(tf.reshape(pred > threshold, [pred.get_shape()[0], 32*32*32]), tf.bool)
    union = tf.cast(tf.reduce_sum(tf.cast(tf.logical_or(gtrs, pred),  tf.int64), axis=1), tf.float32)
    inter = tf.cast(tf.reduce_sum(tf.cast(tf.logical_and(gtrs, pred), tf.int64), axis=1), tf.float32)
    return inter / union


def maxIoU_tf(gtrs_vox, pred_vox, step=1e-1):
    ts = np.arange(0., 1., step)
    ious = []
    for t in ts:
        iou = tf.expand_dims(iou_t_tf(gtrs_vox, pred_vox, threshold=t), 0)
        ious.append(iou)

    ious = tf.concat(ious, axis=0)
    return tf.reduce_mean(tf.reduce_max(ious, axis=0))

