from __future__ import division
import time
from torch.utils.data import Subset
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet_test import MyDarknet
import pickle as pkl
import pandas as pd
import random
import albumentations as A
from custom_coco_c import CIOU_xywh_torch
from torch.nn.utils.rnn import pad_sequence
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
from copy import copy
from copy import deepcopy
import torch.nn.functional as F
from train import train_model, evaluate_model
from custom_coco import CustomCoco, calculate_APs
import matplotlib.pyplot as plt

# Set device to GPU or CPU
gpu = "1"
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()  # Frees up unused GPU memory

train_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#path2data_train="/root/COCO/train2017"
#path2json_train="/root/COCO/annotations/instances_train2017.json"

path2json_val = "/home/jupyter-st124872/fiftyone/coco-2017/raw/instances_val2017.json"
path2data_val = "/home/jupyter-st124872/fiftyone/coco-2017/validation"





train_transform = A.Compose([
    A.SmallestMaxSize(256),
    A.RandomCrop(width=224, height=224),
    # A.HorizontalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

eval_transform = A.Compose([
    A.SmallestMaxSize(256),
    A.CenterCrop(width=224, height=224),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

# raw_train_dataset = torchvision.datasets.CocoDetection(root = path2data_train,
                                # annFile = path2json_train, transform=none_train_transform)

# train_dataset = torchvision.datasets.CocoDetection(root = path2data_train,
                                # annFile = path2json_train, transform=train_transform)
BATCH_SIZE = 1
val_dataset = Subset(CustomCoco(root = path2data_val,
                                annFile = path2json_val, transform=eval_transform), list(range(0,20)))

def collate_fn(batch):
    """ ✅ Removes any missing samples before training. """
    batch = [b for b in batch if b is not None]  
    if len(batch) == 0:
        return None  # ✅ Avoids returning empty batches
    return tuple(zip(*batch))

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=1, collate_fn=collate_fn)

#If there's a GPU availible, put the model on GPU
# CUDA = torch.cuda.is_available()

import torch
torch.cuda.empty_cache() 

print("Loading network.....")
model = MyDarknet("cfg/yolov4.cfg")
model.load_weights("csdarknet53-omega_final.weights", backbone_only=True)
print("Network successfully loaded")


model.to(device)

criterion = nn.CrossEntropyLoss()
params_to_update = model.parameters()
optimizer = optim.Adam(params_to_update, lr=0.001)
for e in range(0, 40):
    running_loss = 0.0
    for inputs, labels, bboxes in val_dataloader:
        inputs = torch.from_numpy(np.array(inputs)).squeeze(1).permute(0,3,1,2).float()
        inputs = inputs.to(device)
        labels = torch.stack(labels).to(device)
        
        running_corrects = 0

        # zero the parameter gradients
        # it uses for update training weights
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs, True)
            # pred_xy = outputs[..., :2] / 224
            # pred_wh = torch.sqrt(outputs[..., 2:4] / 224)

            pred_xywh = outputs[..., 0:4] / 224
            # pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
            pred_conf = outputs[..., 4:5]
            pred_cls = outputs[..., 5:]


            # label_xy = labels[..., :2] / 224
            # label_wh = torch.sqrt(labels[..., 2:4] / 224)

            label_xywh = labels[..., :4] / 224

            # label_xywh = torch.cat([label_xy, label_wh], dim=-1)
            label_obj_mask = labels[..., 4:5]
            label_noobj_mask = (1.0 - label_obj_mask)  # * (
                # iou_max < self.__iou_threshold_loss
            # ).float()
            lambda_coord = 0.001
            lambda_noobj = 0.05
            label_cls = labels[..., 5:]
            loss = nn.MSELoss()
            loss_bce = nn.BCELoss()

            # ciou = CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)

            loss_coord = lambda_coord * label_obj_mask * loss(input=pred_xywh, target=label_xywh)
            loss_conf = (label_obj_mask * loss_bce(input=pred_conf, target=label_obj_mask)) + \
                        (lambda_noobj * label_noobj_mask * loss_bce(input=pred_conf, target=label_obj_mask))
            loss_cls = label_obj_mask * loss_bce(input=pred_cls, target=label_cls)

            loss_coord = torch.sum(loss_coord)
            loss_conf = torch.sum(loss_conf)
            loss_cls = torch.sum(loss_cls)

            # print(pred_xywh.shape, label_xywh.shape)

            ciou = CIOU_xywh_torch(pred_xywh, label_xywh)
            # print(ciou.shape)
            ciou = ciou.unsqueeze(-1)
            # print(ciou.shape)
            # print(label_obj_mask.shape)
            loss_ciou = torch.sum(label_obj_mask * (1.0 - ciou))
            # print(loss_coord)
            loss =  loss_ciou +  loss_conf + loss_cls
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)
            # print('Running loss')
            # print(loss_coord, loss_conf, loss_cls)
    epoch_loss = running_loss / 750
    print(epoch_loss)
    print('Epoch')

    print(calculate_APs(0.5, None, None))
    # break
    # print(x.shape)
    # print(y.shape)
    # print(w.shape)
    # print(h.shape)
    # print(obj.shape)
    # print(cls.shape)
    # break
