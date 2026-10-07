# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import time

import torch
import wandb
from PIL import Image, ImageFile
import errno
import json
import pickle as pkl
import os
import os.path as osp
import yaml
from easydict import EasyDict as edict
import logging
import sys

from utils import misc
import torch.distributed as dist

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def get_text_embedding(path, length):
    with open(path, 'rb') as f:
        word_frequency = pkl.load(f)


def save_train_configs(path, args):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/configs.yaml', 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def load_train_configs(path):
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)


def save_model(name, model, save_path=None, optimizer=None, scheduler=None, epoch=None, logger=None, args=None):
    if not misc.is_main_process():
        return
    if optimizer is None or scheduler is None:
        to_save = model.state_dict()
    else:
        to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }
    save_path = save_path or args.output_dir
    os.makedirs(save_path, exist_ok=True)
    save_file = osp.join(save_path, '{}.pth'.format(name))
    if logger is not None:
        logger.info('Saving checkpoint to {}'.format(save_file))
    torch.save(to_save, save_file)


class LoggerX:
    def __init__(self, args, training=True, add_time=False):
        # if add_time:
        #     cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        #     self.output_dir = osp.join(args.output_dir, args.dataset_name + '_' + args.run_name + '_' + cur_time)
        # else:
        #     self.output_dir = osp.join(args.output_dir, args.dataset_name + '_' + args.run_name)
        self.output_dir = args.output_dir
        self.wandb_enable = args.wandb_enable
        if misc.is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)
            if self.wandb_enable:
                wandb.init(project=args.project_name,
                           name=args.run_name,
                           config=args,
                           settings=wandb.Settings(_disable_stats=True))

            self.logger = logging.getLogger(args.run_name)
            self.logger.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler(stream=sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(
                logging.Formatter("[%(asctime)s.%(msecs)06d] %(levelname)s: %(message)s", datefmt='%H:%M:%S'))
            self.logger.addHandler(console_handler)

            file_name = 'train_log.txt' if training else 'test_log.txt'
            file_handler = logging.FileHandler(osp.join(self.output_dir, file_name), mode='a')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
            self.logger.addHandler(file_handler)

        if misc.is_dist_avail_and_initialized():
            dist.barrier()

    def info(self, msg):
        if misc.is_main_process():
            if isinstance(msg, dict):
                msg = str(msg)
            self.logger.info(msg)

    def debug(self, msg):
        if misc.is_main_process():
            if isinstance(msg, dict):
                msg = str(msg)
            self.logger.debug(msg)

    def wandb_log(self, metric: dict):
        if misc.is_main_process():
            if self.wandb_enable:
                wandb.log(metric)
            output_str = []
            for k, v in metric.items():
                if isinstance(v, int):
                    output_str.append(f"{k}={v}")
                elif isinstance(v, float):
                    output_str.append(f"{k}={round(v, 3)}")
                else:
                    output_str.append(f"{k}: {str(v)}")
            output_str = ' '.join(output_str)
            self.logger.info(output_str)

    def wandb_finish(self):
        if misc.is_main_process() and self.wandb_enable:
            wandb.finish()
