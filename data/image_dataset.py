import functools
import logging
import bisect

import torch.utils.data as data
import cv2
import numpy as np
import glob
from concern.config import Configurable, State
import math
import h5py
import re
import itertools
import os
from torch.utils.data import ConcatDataset
import scipy.io as scio
import json


class hierarchical_dataset(data.Dataset, Configurable):
    """ select_data='/' contains all sub-directory of root directory """

    data_dir = State()
    data_list = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, data_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.data_list = data_list or self.data_list

        dataset_list = []
        select_data = "/"

        #-------------------------------------------------------------------------------------------------------------#

        for dt in self.data_dir:

            if 'SynthText-KR' in dt :
                # syn_kor
                print(f"dataset_root:    {dt}\t dataset: {select_data[0]}")
                for dirpath, dirnames, filenames in os.walk(dt + "/"):
                    for i in filenames:
                        lmdb_path = os.path.join(dirpath, str(i))
                        print(lmdb_path)
                        dataset = ImageDataset_syn(data_dir=[lmdb_path], data_list=self.data_list, cmd=cmd, **kwargs)
                        dataset_list.append(dataset)
            else:
                # syn en
                dataset = ImageDataset_syn_en(data_dir=[dt], data_list=self.data_list, cmd=cmd, **kwargs)
                dataset_list.append(dataset)

        # -------------------------------------------------------------------------------------------------------------#

        self.total_syn_dataset = ConcatDataset(dataset_list)


    def __getitem__(self, index, retry=0):
        data = self.total_syn_dataset[index]
        return data

    def __len__(self):
        return len(self.total_syn_dataset)




class ImageDataset_syn_en(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    data_list = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, data_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.data_list = data_list or self.data_list
        if 'train' in self.data_list[0]:
            self.is_training = True
        else:
            self.is_training = False
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        #self.gt_paths = []
        self.get_all_samples()

    def get_all_samples(self):

        self.get_gt = scio.loadmat(os.path.join(self.data_dir[0], "gt.mat"))

        self.image_paths  = self.get_gt["imnames"][0]
        self.num_samples = len(self.image_paths)
        self.wordBB = self.get_gt["wordBB"][0]
        self.img_words = self.get_gt["txt"][0]


    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}

        path = os.path.join(self.data_dir[0], self.image_paths[index][0])
        image = cv2.imread(path, cv2.IMREAD_COLOR).astype('float32')

        if len(self.wordBB[index].shape) == 3:
            total_bbox = self.wordBB[index].transpose((2, 1, 0))
        else:
            total_bbox = self.wordBB[index].transpose((1, 0))
            total_bbox = np.expand_dims(total_bbox, axis=0)



        # img
        image_path = self.image_paths[index][0]
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = image

        # label

        words = [
            re.split(" \n|\n |\n| ", word.strip()) for word in self.img_words[index]
        ]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]


        line = []
        for i in range(len(words)):
            target = {}
            word_bbox = total_bbox[i]
            word_bbox = np.array(word_bbox).astype(np.float32).tolist()
            target['poly'] = word_bbox
            target['text'] = words[i]
            line.append(target)

        data['lines'] = line
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)

        return data

    def __len__(self):
        return len(self.image_paths)




class ImageDataset_syn(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    data_list = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, data_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.data_list = data_list or self.data_list
        if 'train' in self.data_list[0]:
            self.is_training = True
        else:
            self.is_training = False
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        #self.gt_paths = []
        self.get_all_samples()

    def get_all_samples(self):

        self.get_gt = h5py.File(self.data_dir[0], "r")
        with h5py.File(self.data_dir[0], "r") as file:
            self.image_paths = np.array(list(file["data"].keys()))
            self.num_samples = len(self.image_paths)

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}

        gt = self.get_gt["data"][self.image_paths[index]]
        img = gt[...].astype('float32')

        # img
        image_path = self.image_paths[index]
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img

        # label
        #------------------------------------------------------------------#
        # word 단위 annotation
        # wordBB = gt.attrs["wordBB"]
        # txt = gt.attrs["txt"]
        # all_char_bbox = wordBB.transpose((2, 1, 0))
        # try:
        #     words = [re.split(" \n|\n |\n| ", t.strip()) for t in txt]
        # except:
        #     txt = [t.decode("UTF-8") for t in txt]
        #     words = [re.split(" \n|\n |\n| ", t.strip()) for t in txt]
        #
        # words = list(itertools.chain(*words))
        # words = [t for t in words if len(t) > 0]
        #
        #
        # line = []
        # for i in range(len(words)):
        #     target = {}
        #     word_bbox = all_char_bbox[i]
        #     word_bbox = np.array(word_bbox).astype(np.float32).tolist()
        #     target['poly'] = word_bbox
        #     target['text'] = words[i]
        #     line.append(target)
        # ------------------------------------------------------------------#

        # char 단위 annotation
        charBB = gt.attrs["charBB"]
        txt = gt.attrs["txt"]
        all_char_bbox = charBB.transpose((2, 1, 0))


        try:
            words = [re.split(" \n|\n |\n| ", t.strip()) for t in txt]
        except:
            txt = [t.decode("UTF-8") for t in txt]
            words = [re.split(" \n|\n |\n| ", t.strip()) for t in txt]

        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

        word_level_char_bbox = []
        char_idx = 0
        for i in range(len(words)):
            length_of_word = len(words[i])
            word_bbox = all_char_bbox[char_idx: char_idx + length_of_word]
            assert len(word_bbox) == length_of_word
            char_idx += length_of_word
            word_bbox = np.array(word_bbox)

            xmax =  word_bbox[:,:,0].max()
            xmin = word_bbox[:, :, 0].min()
            ymax = word_bbox[:, :, 1].max()
            ymin = word_bbox[:, :, 1].min()

            new_word_bbox = np.array([
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],

            ])
            #new_word_bbox = np.expand_dims(new_word_bbox, axis=0)
            word_level_char_bbox.append(new_word_bbox)


        line = []
        for i in range(len(words)):
            target = {}
            word_bbox = word_level_char_bbox[i]
            word_bbox = np.array(word_bbox).astype(np.float32).tolist()
            target['poly'] = word_bbox
            target['text'] = words[i]

            line.append(target)

        data['lines'] = line
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)

        return data

    def __len__(self):
        return len(self.image_paths)

class ImageDataset_prc(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    data_list = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, data_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.data_list = data_list or self.data_list
        if 'train' in self.data_list[0]:
            self.is_training = True
        else:
            self.is_training = False
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        self.gt_paths = []
        self.get_all_samples()

    def get_all_samples(self):

        image_path = []
        gt_path = []
        for dirpath, dirnames, filenames in os.walk(self.data_dir[0] + "/"):
            for i in filenames:
                data_path = os.path.join(dirpath, str(i))
                filename, extension = os.path.splitext(data_path)

                if extension == '.jpg':
                    image_path.append(data_path)
                    gt_path.append(filename+'_label.txt')

        self.image_paths = sorted(image_path)
        self.gt_paths = sorted(gt_path)

        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()


    def load_ann(self):
        res = []
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, encoding="utf-8").readlines()
            for line in reader:
                item = {}
                parts = line.strip().encode("utf-8").decode("utf-8-sig").split("##::")
                label = parts[-1]
                if label == 'dnc':
                    label = '###'
                parts_box = parts[0].strip().encode("utf-8").decode("utf-8-sig").split(" ")
                poly = np.array(list(map(float, parts_box[:8]))).reshape((-1, 2)).tolist()

                item['poly'] = poly
                item['text'] = label
                lines.append(item)
            res.append(lines)

        return res

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')

        data['filename'] = image_path.split('/')[-1]
        data['data_id'] = image_path.split('/')[-1]

        data['image'] = img
        target = self.targets[index]
        data['lines'] = target

        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)


        return data

    def __len__(self):
        return len(self.image_paths)



class ImageDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    data_list = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, data_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.data_list = data_list or self.data_list
        if 'train' in self.data_list[0]:
            self.is_training = True
        else:
            self.is_training = False
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        self.gt_paths = []
        self.get_all_samples()

    def get_all_samples(self):
        for i in range(len(self.data_dir)):

            filename, extension = os.path.splitext(self.data_list[i])
            if extension == '.txt':
                with open(self.data_list[i], 'r') as fid:
                    image_list = fid.readlines()
            elif extension == '.json':
                with open(self.data_list[i], "r") as fid:
                    data = json.load(fid)
                image_list = list(data['annots'].keys())
            else:
                print('not data list')


            if self.is_training:
                image_path=[self.data_dir[i]+'/train_images/'+timg.strip() for timg in image_list]
                gt_path=[self.data_dir[i]+'/train_gts/'+timg.strip()+'.txt' for timg in image_list]
            else:
                image_path=[self.data_dir[i]+'/test_images/'+timg.strip() for timg in image_list]
                print(self.data_dir[i])
                if 'TD500' in self.data_list[i] or 'total_text' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/test_gts/'+timg.strip()+'.txt' for timg in image_list]
                else:
                    gt_path=[self.data_dir[i]+'/test_gts/'+'gt_'+timg.strip().split('.')[0]+'.txt' for timg in image_list]
            self.image_paths += image_path
            self.gt_paths += gt_path
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def load_ann(self):
        res = []
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, 'r').readlines()
            for line in reader:
                item = {}
                parts = line.strip().split(',')
                label = parts[-1]
                if 'TD' in self.data_dir[0] and label == '1':
                    label = '###'
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                if 'icdar2015' in self.data_dir[0]:
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                elif 'icdar2013' in self.data_dir[0]:
                    box = line[:4]
                    poly = [
                        [float(box[0]), float(box[1])],
                        [float(box[2]), float(box[1])],
                        [float(box[2]), float(box[3])],
                        [float(box[0]), float(box[3])],
                    ]

                else:
                    num_points = math.floor((len(line) - 1) / 2) * 2
                    poly = np.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
                item['poly'] = poly
                item['text'] = label
                lines.append(item)
            res.append(lines)
        return res

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')

        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        target = self.targets[index]
        data['lines'] = target

        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        return data

    def __len__(self):
        return len(self.image_paths)
