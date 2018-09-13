import tensorflow as tf

def batch(ds, bs):
    """[ds] dataset; [bs] batch-size"""
    return ds.apply(tf.contrib.data.batch_and_drop_remainder(bs))

