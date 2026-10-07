import os.path as op
from typing import List

import torch

from utils.iotools import read_json
from torch.utils.data import Dataset
import scipy.io as sio


class InteractiveReIDDataset(Dataset):
    """
    annotation format:
    [{'split': str,
      'file_path': str,
      'fine-grained_caption': str,
      'vague_caption': str,
      'coarse-grained_caption': str,
      'id': int}...]
    """
    pid_offset = {'CUHK-PEDES': 0,
                  'ICFG-PEDES': 0,
                  'RSTPReid': 0}

    def __init__(self, root='', verbose=False):
        super(InteractiveReIDDataset, self).__init__()

        self.dataset_dir = root
        self.img_dir = root
        self.pid_offset = 0
        self.anno_path = op.join(self.dataset_dir, "Interactive-PEDES_interactive_annos.json")

        print("loading annotation from {}.".format(self.anno_path))
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)
        print('len train_annos:', len(self.train_annos))
        print('len test_annos:', len(self.test_annos))
        print('len val_annos:', len(self.val_annos))

        self.human_anno_training = False
        self.human_anno_testing = False
        self.use_vision_answer = False

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        # self.train, self.train_id_container = self._process_anno(self.test_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            print("=> Interactive Images and Captions are loaded")

    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id']) + self.pid_offset  # make pid begin from 0
                pid_container.add(pid)
                img_path = anno['file_path']
                questions = anno['questions']
                answers = anno['answers']
                # vague_caption = anno['vague_caption']
                if self.human_anno_training:
                    fine_grained_caption = [anno['fine-grained_description'], anno['captions'][0]]
                else:
                    fine_grained_caption = [anno['fine-grained_description']]  # caption list

                if self.human_anno_testing:
                    initial_description = anno['captions'][0]
                else:
                    initial_description = anno['initial_description']

                for f_cap in fine_grained_caption:
                    dataset.append((pid, image_id, img_path, f_cap, initial_description, questions, answers))

                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            img_paths = []
            image_pids = []
            fine_grained_captions = []
            initial_description = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                img_path = op.join(self.img_dir, anno['file_path'])

                if self.human_anno_testing:
                    fine_cap = anno['fine-grained_description']
                    init_cap = anno['captions'][:1]
                else:
                    fine_cap = anno['fine-grained_description']
                    init_cap = [anno['initial_description']]

                if self.use_vision_answer:
                    fine_cap = anno['image_path']

                image_pids.append(pid)
                img_paths.append(img_path)
                pid_container.add(pid)

                for i_cap in init_cap:
                    caption_pids.append(pid)
                    fine_grained_captions.append(fine_cap)
                    initial_description.append(i_cap)

            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "fine-grained_description": fine_grained_captions,
                "initial_description": initial_description
            }
            return dataset, pid_container

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
