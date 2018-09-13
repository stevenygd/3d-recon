import os
import cv2
import json
import numpy as np
from data_loaders import binvox_rw
import itertools
from random import shuffle
from random import choice
import tensorflow as tf
from data_loaders.util import batch
dir_path = os.path.dirname(os.path.realpath(__file__))

SHAPENET_VOX_PATH = os.path.join(dir_path, "..", "ShapeNetVox32")

# Streaming into the memory.
class StreamingValDataLoader(object):

    def __init__(self, data_settings, shape=[32,32,1], viewpoints=10, binarize_data=True):
        self.shape = shape[:3]
        self.viewpoints = viewpoints
        self.binarize_data = binarize_data

        self.pose_info = {}
        self.imgs = []
        for data_dir in data_settings:
            pose_info = json.load(open(os.path.join(data_dir, "pose_info.json")))
            self.pose_info.update(pose_info)
            self.imgs += [(data_dir, fname) for fname in pose_info.keys()]

        def _gen_():
            shuffle(self.imgs)
            for data_dir, fname in self.imgs:
                vox = self._fname_to_vox_(fname).astype(np.float32)
                pose = self._fname_to_pose_(fname).astype(np.float32)
                img = self._fname_to_img_(data_dir, fname).astype(np.float32)
                yield {
                    "image" : img,
                    "vox"   : vox,
                    "pose"  : pose
                }

        self.dataset = tf.data.Dataset.from_generator(
            _gen_, {
                "image" : tf.float32,
                "pose"  : tf.float32,
                "vox"   : tf.float32
            }, output_shapes={
                "image" : self.shape,
                "pose"  : (3,),
                "vox"   : (32,32,32)
            })

    def _fname_to_pose_(self, fname):
        return np.array([
            self.pose_info[fname]['rx'],
            self.pose_info[fname]['ry'],
            self.pose_info[fname]['rz']
        ]).astype(np.float32)

    def _fname_to_vox_(self, fname):
        """fname in pose_info.keys()"""
        sid, mid = fname.split("_")[:2]
        # k = "%s_%s"%(sid, mid)
        ret = None
        with open(os.path.join(SHAPENET_VOX_PATH, sid, mid, "model.binvox"), "rb") as f:
            md = binvox_rw.read_as_3d_array(f)
            ret = md.data.astype(np.float32)
        return ret

    def _fname_to_img_(self, data_dir, fname):
        img = cv2.imread(os.path.join(data_dir, fname), 0)[..., np.newaxis].astype(np.float32)
        if self.binarize_data:
            img = (img > 0).astype(np.float32) * 255
        img /= 255.
        return img

    def __call__(self, batch_size):
        iterator = batch(self.dataset, batch_size).make_initializable_iterator()
        return iterator.initializer, iterator.get_next()

    def __len__(self):
        return len(self.imgs)


class StreamingSplitDataLoader(object):

    """Will split the datasets into two halves, one sample with Pos the other witout
       build for GAN-AutoEncoder
    """
    def __init__(self, data_setting, shape=[32,32,1], viewpoints=10, binarize_data=False,
                 disjoint_split=False):
        self.shape = shape[:3]
        self.viewpoints = viewpoints
        self.binarize_data = binarize_data

        self.pose_info = {}           # abs path -> pose information
        self.modelid_withpose = set() # all model id for pos supervision
        self.modelid_nopose   = set() # all model id for non-pose supervision
        self.models = {} # mode_id -> list of images in absolute path
        for data_dir, p_ratio, nop_ratio in data_setting:
            # Update pose information
            pose_info = json.load(open(os.path.join(data_dir, "pose_info.json")))
            self.pose_info.update({
                os.path.join(data_dir, k):pose_info[k] for k in pose_info.keys()
            })

            # get model keys from this data_dir
            model_keys = set()
            for fname in pose_info.keys():
                sid, mid = fname.split("_")[:2]
                model_id = "%s-%s"%(sid, mid)
                model_keys.add(model_id)
                if model_id not in self.models:
                    self.models[model_id] = []
                if len(self.models[model_id]) < self.viewpoints:
                    self.models[model_id].append(os.path.join(data_dir, fname))
            model_keys = list(model_keys)

            # splits
            n = int(len(model_keys)*p_ratio)
            m = int(len(model_keys)*nop_ratio)


            for mid in model_keys[:n]:
                self.modelid_withpose.add(mid)

            if disjoint_split:
                assert n + m <= len(model_keys)
                for mid in model_keys[n:n+m]:
                    self.modelid_nopose.add(mid)
            else:
                for mid in model_keys[:m]:
                    self.modelid_nopose.add(mid)

        self.vp_pairs = itertools.product(range(self.viewpoints), range(self.viewpoints))
        self.vp_pairs = [(i,j) for i,j in self.vp_pairs if not (i > j)]

        self.modelid_withpose = list(self.modelid_withpose)
        self.modelid_nopose   = list(self.modelid_nopose)

        print("Dataset with pose:%d"%len(self.modelid_withpose))
        print("Dataset no pose  :%d"%len(self.modelid_nopose))

        def _gen_encdec_():
            while True:
                shuffle(self.modelid_withpose)
                for mid in self.modelid_withpose:
                    i, j = choice(self.vp_pairs)
                    fname_i, fname_j = self.models[mid][i], self.models[mid][j]
                    img_i, img_j = self._fname_to_img_(fname_i), self._fname_to_img_(fname_j)
                    pos_i, pos_j = self._fname_to_pose_(fname_i), self._fname_to_pose_(fname_j)
                    yield {
                        "img_1"  : img_i,
                        "img_2"  : img_j,
                        "pos_1"  : pos_i,
                        "pos_2"  : pos_j
                    }

        self.encdec_dataset = tf.data.Dataset.from_generator(
                _gen_encdec_, {
                    "img_1" : tf.float32, "img_2" : tf.float32,
                    "pos_1" : tf.float32, "pos_2" : tf.float32
            }, output_shapes={
                    "img_1" : (32, 32, 1), "img_2" : (32, 32, 1),
                    "pos_1" : (3,), "pos_2" : (3,)
            }).repeat().shuffle(max(1, len(self.modelid_withpose))) # TODO: magic number

        def _gen_gan_():
            while True:
                shuffle(self.modelid_nopose)
                for mid in self.modelid_nopose:
                    for i in range(self.viewpoints):
                        img = self._fname_to_img_(self.models[mid][i])
                        yield {"img" : img}

        self.gan_dataset = tf.data.Dataset.from_generator(
                _gen_gan_, {
                    "img" : tf.float32
                }, output_shapes={
                    "img" : (32, 32, 1)
            }).repeat().shuffle(max(1, len(self.modelid_nopose))) # TODO: magic number

    def _fname_to_img_(self, fname):
        img = cv2.imread(fname, 0)[..., np.newaxis].astype(np.float32)
        if self.binarize_data:
            img = (img > 0).astype(np.float32) * 255
        img /= 255.
        return img

    def _fname_to_pose_(self, fname):
        return np.array([
            self.pose_info[fname]['rx'],
            self.pose_info[fname]['ry'],
            self.pose_info[fname]['rz']
        ]).astype(np.float32)

    def __call__(self, bs, sample_type='both'):
        encdec_next = batch(self.encdec_dataset, bs).make_one_shot_iterator().get_next()
        gan_next    = batch(self.gan_dataset, bs).make_one_shot_iterator().get_next()
        return encdec_next, gan_next

    def __len__(self):
        n_pose = len(self.modelid_nopose)*self.viewpoints
        n_nopose = len(self.modelid_withpose)*len(self.vp_pairs)
        return n_pose, n_nopose

class NoiseSampler(object):
    def __init__(self, use_normal=False):
        self.use_normal = use_normal
    def __call__(self, batch_size, z_dim):
        if self.use_normal:
            vp = np.random.uniform(-1.0, 1.0, [batch_size, 1])
            other = np.random.normal(size=[batch_size, z_dim-1])
            ret = np.concatenate([other,vp], axis=1)
        else:
            ret = np.random.uniform(-1.0, 1.0, [batch_size, z_dim])
        return ret.astype(np.float32)

class MultiViewNoiseSampler(object):
    def __init__(self, num_viewpoints=10):
        self.num_viewpoints = num_viewpoints

    def __call__(self, batch_size, z_dim):
        noise = np.random.uniform(-1.0, 1.0, [batch_size//self.num_viewpoints+1, z_dim-1])
        out = np.zeros((batch_size, z_dim))
        for i in range(batch_size):
            out[i,:-1] = noise[i//self.num_viewpoints,...]
            out[i,-1]  = -1 + 2./self.num_viewpoints*(i%self.num_viewpoints)
        return out.astype(np.float32)


if __name__ == "__main__":
    ds = StreamingValDataLoader(
            ["../data/airplanes_32x32alpha_perspective_d100_r32_vp24_discrete_PTN/train"],
            viewpoints=8)
    ds_init, ds_batch = ds(256)
    print(ds_batch)
    with tf.Session() as sess:
        sess.run(ds_init)
        try:
            for i in range(100):
                ret = sess.run(ds_batch)
        except tf.errors.OutOfRangeError:
            print("End of dataset")  # ==> "End of dataset"
        for k in ret.keys():
            v = ret[k]
            print(k + ":" + str(v.shape))

    ds = StreamingSplitDataLoader([
        ("../data/airplanes_32x32alpha_perspective_d100_r32_vp24_discrete_PTN/train", 0.5),
        ("../data/chairs_32x32alpha_perspective_d100_r32_vp24_discrete_PTN/train", 0.5),
        ("../data/benches_32x32alpha_perspective_d100_r32_vp24_discrete_PTN/train", 0.5),
    ], viewpoints=8)
    encdec_batch, gan_batch = ds(32)
    print(encdec_batch)
    print(gan_batch)
    with tf.Session() as sess:
        for i in range(10):
            ret = sess.run(encdec_batch)
            ret.update(sess.run(gan_batch))
        for k in ret.keys():
            v = ret[k]
            print(k + ":" + str(v.shape))

    pass
