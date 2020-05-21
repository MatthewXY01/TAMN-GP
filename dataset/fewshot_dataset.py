import os
from PIL import Image
import numpy as np
import os.path as osp
import io
import random

import torch
from torch.utils.data import Dataset
torch.set_printoptions(threshold=200)
class FewShotDataset_train(Dataset):
    """
    Build taskset based on 'train' split of the standard source dataset.
    Returns a task (Xspt, Yspt, Xqry, Yqry, Ycls) to classify'
        Xspt: [N_way*K_shot, c, h, w].
        Yspt: [N_way*K_shot].
        Xqry: [N_way*num_query, c, h, w].
        Yqry: [N_way*num_query].
        Ycls: [N_way*num_query]. 
    """

    def __init__(self, 
                datapoints,
                 labels2idx, # labels of index {(cats: index1, index2, ...)}.
                 labelIDs, # train labels [0, 1, 2, 3, ...,].
                 N_way=5, # number of novel categories.
                 K_shot=1, # number of training examples per novel category.
                 nTestNovel=6*5, # number of test examples for all the novel categories.
                 epoch_size=2000, # number of tasks per eooch.
                 transform=None,
                 **kwargs
                 ):
        self.datapoints = datapoints
        self.labels2idx = labels2idx
        self.labelIDs = labelIDs
        self.N_way = N_way
        self.K_shot = K_shot
        self.transform = transform
        self.nTestNovel = nTestNovel
        self.epoch_size = epoch_size

        # datapoints [(dir1, 0), (dir2,0),..., (dir38400, 63)] len: 38400
        # labels2idx {0:[0, 1,..., 599], 1:[600, 601,..., 1199],..., 63:[37800, 37801,..., 38399]} len: 64
        # labelIDs [0, 1, 2,..., 63] len: 64
    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        """
        Sample an episode in the form of the img indexs.
        Returns:

        """
        cls_ids = random.sample(self.labelIDs, self.N_way)
        assert(((self.nTestNovel % self.N_way)==0) and len(cls_ids)==self.N_way)
        nQueryPerClass = int(self.nTestNovel/self.N_way)

        Support = []
        Query = []

        for cls in range(len(cls_ids)):
            num_img = nQueryPerClass + self.K_shot
            img_ids = random.sample(self.labels2idx[cls_ids[cls]], num_img) 

            spt_ids = img_ids[:self.K_shot]
            qry_ids = img_ids[self.K_shot:]

            Support += [(img_id, cls) for img_id in spt_ids]
            Query += [(img_id, cls) for img_id in qry_ids]
        assert(len(Support) == self.K_shot*self.N_way)
        assert(len(Query) == self.nTestNovel)
        # random.shuffle(Support)
        # random.shuffle(Query)

        return Support, Query
    
    def _create_tensor(self, examples):
        """
        Creats the examples image label tensor data.

        Args:
            examples: a list of 2-element tuples. (sample_index, label).

        Returns:
            images: a tensor [N_way*K_shot, c, h, w]
            labels: a tensor [N_way*K_shot] with elements in {0, 1,..., N_way-1}
            cls: a tensor [N_way*K_shot] with elements in {0, 1,.., 63}(if smaples are from training set)
        """
        images = []
        labels = []
        cls = []

        for (img_idx, label) in examples:
            img_path, idx = self.datapoints[img_idx]
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
            labels.append(label)
            cls.append(idx)

        images = torch.stack(images, dim = 0)
        labels = torch.LongTensor(labels)
        cls = torch.LongTensor(cls)
        return images, labels, cls
    
    def __getitem__(self, ind):
        Support, Query = self._sample_episode()
        Xspt, Yspt, _ = self._create_tensor(Support)
        Xqry, Yqry, Zqry = self._create_tensor(Query) # Zqry 是query图片在全部类别（训练集有64类）中的编号
        return Xspt, Yspt, Xqry, Yqry, Zqry

class FewShotDataset_eval(Dataset):
    """
    Build taskset based on 'val' or 'test' split of the standard source dataset.
    Returns a task (Xspt, Yspt, Xqry, Yqry, Ycls) to classify'
        Xspt: [N_way*K_shot, c, h, w].
        Yspt: [N_way*K_shot].
        Xqry: [N_way*num_query, c, h, w].
        Yqry: [N_way*num_query].
        Ycls: [N_way*num_query]. 
    """

    def __init__(self, 
                datapoints,
                 labels2idx, # labels of index {(cats: index1, index2, ...)}.
                 labelIDs, # train labels [0, 1, 2, 3, ...,].
                 N_way=5, # number of novel categories.
                 K_shot=1, # number of training examples per novel category.
                 nTestNovel=15*5, # number of test examples for all the novel categories.
                 epoch_size=2000, # number of tasks per eooch.
                 transform=None,
                 **kwargs
                 ):
        self.datapoints = datapoints
        self.labels2idx = labels2idx
        self.labelIDs = labelIDs
        self.N_way = N_way
        self.K_shot = K_shot
        self.transform = transform
        self.nTestNovel = nTestNovel
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        """
        Sample an episode in the form of the img indexs.
        Returns:

        """
        cls_ids = random.sample(self.labelIDs, self.N_way)
        assert(((self.nTestNovel % self.N_way)==0) and len(cls_ids)==self.N_way)
        nQueryPerClass = int(self.nTestNovel/self.N_way)

        Support = []
        Query = []

        for cls in range(len(cls_ids)):
            num_img = nQueryPerClass + self.K_shot
            img_ids = random.sample(self.labels2idx[cls_ids[cls]], num_img) 

            spt_ids = img_ids[:self.K_shot]
            qry_ids = img_ids[self.K_shot:]

            Support += [(img_id, cls) for img_id in spt_ids]
            Query += [(img_id, cls) for img_id in qry_ids]
        assert(len(Support) == self.K_shot*self.N_way)
        assert(len(Query) == self.nTestNovel)

        return Support, Query
    
    def _create_tensor(self, examples):
        """
        Creats the examples image label tensor data.

        Args:
            examples: a list of 2-element tuples. (sample_index, label).

        Returns:
            images: a tensor [N_way*K_shot, c, h, w]
            labels: a tensor [N_way*K_shot] with elements in {0, 1,..., N_way-1}
            cls: a tensor [N_way*K_shot] with elements in {0, 1,.., 15}(if smaples are from validation set)
        """
        images = []
        labels = []
        cls = []

        for (img_idx, label) in examples:
            img_path, idx = self.datapoints[img_idx]
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
            labels.append(label)
            cls.append(idx)

        images = torch.stack(images, dim = 0)
        labels = torch.LongTensor(labels)
        cls = torch.LongTensor(cls)
        return images, labels, cls
    
    def __getitem__(self, ind):
        Support, Query = self._sample_episode()
        Xspt, Yspt, _ = self._create_tensor(Support)
        Xqry, Yqry, Zqry = self._create_tensor(Query)
        return Xspt, Yspt, Xqry, Yqry, Zqry