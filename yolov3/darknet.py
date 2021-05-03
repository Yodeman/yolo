import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgFile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block on the neural
    network to be built. Block is represented as a dictionary in the list.
    """
    
    file = open(cfgFile, 'r')
    lines = file.readlines()
    lines=[x.strip(" \n") for x in lines if len(x)>0 and x[0] != '#']
    
    block= {}
    blocks =[]
    
    for line in lines:
        if line.startswith('['):
            if len(block):
                blocks.append(block)
                block = {}
            block['type'] = line.strip(" []")
        elif line:
            key, value = line.split('=')
            block[key.rstrip()]=value.lstrip()
    blocks.append(block)
    return blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        if (x['type']=="convolutional"):
            #Get info about the layer
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias=True
            except KeyError:
                batch_normalize = 0
                bias=True
                
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1)//2
            else:
                pad = 0
            
            #add convolution layer
            conv = nn.Conv2d(
                in_channels=prev_filters, out_channels=filters, kernel_size=kernel_size, 
                stride=stride, padding=pad, bias=bias
            )
            module.add_module("conv_{0}".format(index), conv)
            
            #add the batch normalization layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
                
            #add leaky ReLU if it is the activation
            if activation=="leaky":
                act=nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), act)
                
        elif (x["type"]=="upsampling"):
            stride = x["stride"]
            upsampling = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsampling)
            
        elif (x["type"]=="route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
                
            if start > 0:
                start = start - index
            if end:
                end = end-index
            
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index+start]+output_filters[index+end]
            else:
                filters = output_filters[index+start]
                
        elif (x["type"]=="shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        elif (x["type"]=="yolo"):
            mask = [int(i) for i in x["mask"].split(",")]
            anchors = list(map(int, x["anchors"].split(",")))
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("detection_{}".format(index), detection)
            
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info, module_list)
        
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self,net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, CUDA):
        """
        x: input image.
        CUDA: If True, the model use GPU else, CPU
        """
        modules = self.blocks[1:]
        outputs = {} # cache the outputs of the route layer
        write = 0
        for i, module in enumerate(modules):
            module_type = module['type']
            if module_type in ("convolutional", "upsample"):
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                
                if (layers[0]) > 0:
                    layers[0] -= i
                    
                if len(layers)==1:
                    x = outputs[i + layers[0]]
                    
                else :
                    if layers[1]>0:
                        layers[1] -= i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    
                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
                
            

if __name__ == "__main__":
    from pprint import pprint
    file = r"\\wsl$\Ubuntu-18.04\home\paul\darknet\cfg\yolov3.cfg"
    blocks = parse_cfg(file)
    net_info, module = create_modules(blocks)
    pprint(net_info)
    print()
    pprint(module)
    