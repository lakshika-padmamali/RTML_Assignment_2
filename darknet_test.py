from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from mish import Mish
from util import *

def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0 and x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()     
        else:
            key, value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]     
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
    
        if (x["type"] == "convolutional"):
            activation = x["activation"]
            batch_normalize = int(x.get("batch_normalize", 0))
            bias = not batch_normalize
        
            filters= int(x["filters"])
            
            padding = (int(x["size"]) - 1) // 2 if int(x["pad"]) else 0
            prev_filters = output_filters[-1] if output_filters else 3  # Ensure correct input channels

            
            conv = nn.Conv2d(prev_filters, filters, int(x["size"]), int(x["stride"]), padding, bias=bias)
            module.add_module("conv_{0}".format(index), conv)
        
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            if activation == "leaky":
                module.add_module("leaky_{0}".format(index), nn.LeakyReLU(0.1, inplace=True))
            elif activation == "mish":
                module.add_module("mish_{0}".format(index), Mish())
        
        elif (x["type"] == "upsample"):
            module.add_module("upsample_{}".format(index), nn.Upsample(scale_factor=2, mode="nearest"))
                
        elif (x["type"] == "route"):
            layers = list(map(int, x["layers"].split(',')))
            filters = sum([output_filters[i] for i in layers])
            module.add_module("route_{0}".format(index), EmptyLayer())
    
        elif x["type"] == "shortcut":
            module.add_module("shortcut_{}".format(index), EmptyLayer())
            
        elif x["type"] == "yolo":
            mask = list(map(int, x["mask"].split(",")))
            anchors = list(map(int, x["anchors"].split(",")))
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            module.add_module("Detection_{}".format(index), DetectionLayer(anchors))
                              
        module_list.append(module)
        output_filters.append(filters)
        prev_filters = filters
        
        
    return (net_info, module_list)

class MyDarknet(nn.Module):
    def __init__(self, cfgfile):
        super(MyDarknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.net_info["height"] = 608

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # Store outputs for route layers
        write = 0

        for i, module in enumerate(modules):
            module_type = module["type"]

            if module_type in ["convolutional", "upsample"]:
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                print(f"🔍 Debugging Route Layer: {layers}")

                try:
                    # ✅ Ensure correct layer indices, including negatives
                    layers = [int(l) if int(l) >= 0 else i + int(l) for l in layers]
                except ValueError:
                    print(f"❌ Error parsing layers: {layers}. Skipping this route layer.")
                    continue

                # ✅ Ensure only valid layers are used
                valid_layers = [outputs[l] for l in layers if l in outputs]

                if not valid_layers:
                    print(f"⚠️ Warning: No valid layers found! Skipping route layer.")
                    continue

                # ✅ Fix: Ensure all layers have the same number of channels
                channel_counts = [v.shape[1] for v in valid_layers]
                if len(set(channel_counts)) > 1:
                    print(f"❌ Mismatched channel sizes in route layers: {channel_counts}. Fixing...")
                    min_channels = min(channel_counts)
                    valid_layers = [v[:, :min_channels, :, :] for v in valid_layers]  # Trim to smallest channel count
                
                if len(valid_layers) > 1:
                    print(f"✅ Valid Route Layers: {[v.shape for v in valid_layers]}")
                    x = torch.cat(valid_layers, 1)
                else:
                    x = valid_layers[0]  # Use single valid layer directly

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == "yolo":
                detections = x if not write else torch.cat((detections, x), 1)
                write = 1

            outputs[i] = x  # Store output for future route layers

        return detections
    def load_weights(self, weightfile, backbone_only=False):
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i, block in enumerate(self.blocks[1:]):
            if block["type"] == ["convolutional"]:
                model = self.module_list[i]
                batch_normalize = int(block.get("batch_normalize", 0))
                conv = model[0]
                if batch_normalize:
                    bn = model[1]
                    for param in [bn.bias, bn.weight, bn.running_mean, bn.running_var]:
                        num = param.numel()
                        param.data.copy_(torch.from_numpy(weights[ptr:ptr+num]).view_as(param))
                        ptr += num
                num = conv.weight.numel()
                conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num]).view_as(conv.weight))
                ptr += num
            if backbone_only and block["type"] == ["yolo"]:
                break
