import os
import time
import shutil
import argparse

from lib import model_G, model_D, model_E
from net import DCGANCycEncDecRandomVP
file_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(os.path.join(file_path, 'data'))

from data_loaders import data_loader
from config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='chairs_pose0.5')
    parser.add_argument('--prefix', type=str, default='test')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num-batches', type=int, default=-1)
    args = parser.parse_args()

    cfg = Config()

    data_setting = cfg.category_settings[args.data]
    data_train = data_setting['train']
    data_val   = data_setting['val']
    # learning strategy
    if 'e_lr' in data_setting:
        cfg.e_lr = data_setting['e_lr']
    if 'd_lr' in data_setting:
        cfg.d_lr = data_setting['d_lr']
    if 'g_lr' in data_setting:
        cfg.g_lr = data_setting['g_lr']
    if 'decay_steps' in data_setting:
        cfg.decay_steps = data_setting['decay_steps']
    if 'decay_rate' in data_setting:
        cfg.decay_rate = data_setting['decay_rate']

    # validation / save interval
    if 'validation_interval' in data_setting:
        cfg.validation_interval = data_setting['validation_interval']
    if 'batch_size' in data_setting:
        cfg.batch_size = data_setting['batch_size']
    if 'num_batches' in data_setting and args.num_batches < 0:
        args.num_batches =data_setting['num_batches']

    if 'update_g' in data_setting:
        cfg.update_g = data_setting['update_g']
    if 'update_d' in data_setting:
        cfg.update_d = data_setting['update_d']
    if 'update_e' in data_setting:
        cfg.update_e = data_setting['update_e']
    if 'num_gan_iters' in data_setting:
        cfg.num_gan_iters = data_setting['num_gan_iters']
    if 'num_autoencoder_iters' in data_setting:
        cfg.num_autoencoder_iters = data_setting['num_autoencoder_iters']
    if "batch_size" in data_setting:
        cfg.batch_size = data_setting['batch_size']

    train_setting = []
    for cate, split, p_ratio, nop_ratio in data_train:
        p = os.path.join(dir_path, "data",
            "%s_32x32alpha_perspective_d100_r32_vp24_random_default"%cate, split)
        train_setting.append((p, p_ratio, nop_ratio))
    xs = data_loader.StreamingSplitDataLoader(train_setting,
            shape=[cfg.resolution, cfg.resolution, 1],
            viewpoints=cfg.viewpoints, binarize_data=cfg.binarize_data)

    val_setting = []
    for cate, split in data_val:
        p = os.path.join(dir_path, "data",
            "%s_32x32alpha_perspective_d100_r32_vp24_random_default"%cate, split)
        val_setting.append(p)
    vs = data_loader.StreamingValDataLoader(val_setting,
        shape=[cfg.resolution, cfg.resolution, 1],
        viewpoints=cfg.viewpoints,
        binarize_data=cfg.binarize_data)

    if cfg.noise == 'multi_view':
        zs = data_loader.MultiViewNoiseSampler()
    else:
        zs = data_loader.NoiseSampler(use_normal=cfg.use_normal_noise)

    d_net = model_D.Discriminator(cfg)
    g_net = model_G.Generator(cfg)
    e_net = model_E.Encoder(cfg)

    prefix = os.path.join(args.data, "time%d_%s" % (int(time.time()), args.prefix))
    model_instance = DCGANCycEncDecRandomVP(g_net, e_net, d_net, xs, zs, vs, prefix, cfg)

    print("Resume:%s"%args.resume)
    log_path = model_instance.log_dir
    if not os.path.isdir(log_path):
        raise Exception("Log path :%s doesn't exists"%log_path)

    # Save codes to make experiment reproduciable
    shutil.copyfile('train.py',
                    os.path.join(log_path, 'train.py'))
    shutil.copyfile('net.py',
                    os.path.join(log_path, 'net.py'))
    shutil.copyfile('utils.py',
                    os.path.join(log_path, 'utils.py'))
    shutil.copyfile('training_settings.py',
                    os.path.join(log_path, 'training_settings.py'))
    shutil.copyfile('config.py',
                    os.path.join(log_path, 'config.py'))
    shutil.copytree('lib', os.path.join(log_path, "lib"))
    shutil.copytree('data_loaders', os.path.join(log_path, "data_loaders"))

    model_instance.train(
            num_batches=args.num_batches, resume=args.resume,
            num_autoencoder_iters=cfg.num_autoencoder_iters,
            num_gan_iters=cfg.num_gan_iters,
            max_validation_batches=cfg.max_validation_batches,
            update_d=cfg.update_d, update_g=cfg.update_g, update_e=cfg.update_e
    )
