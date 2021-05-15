import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from darknet import Darknet
import numpy as np
import cv2
import argparse
import pickle as pkl
import pandas as pd
import random
import os

def arg_parse():
    """
    Parse arguments to the detect module
    """
    parser = argparse.ArgumentParser(description="YOLO v3 Detection module")
    parser.add_argument("--images", dest="images", help="Image file or image directory to peform detction on",
                        default="test_img", type=str)
    parser.add_argument("--dest", dest="dest", help="Destination to store detections", default="dest", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object confidence used to filter predictions",
                        default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshold", default=0.4)
    parser.add_argument("--cfg", dest="cfgfile", help="YOLO configuration file", default="./cfg/yolov3.cfg",
                        type=str)
    parser.add_argument("--weights", dest="weightsfile", help="The model trained weights", default="yolov3.weights",
                        type=str)
    parser.add_argument("--reso", dest="reso",
                        help="Input resolution to the network. Increase-->increase accuracy, Decrease-->increase speed.",
                        default="416", type=str)
    
    return parser.parse_args()

def load_classes(path):
    classes = [i.strip() for i in open(path, 'r').readlines() if i!="\n"]
    return classes



args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco_classes.txt")

print("Loading network...")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile) 
print("Network loaded successfully")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert (inp_dim%32 == 0) and (inp_dim > 32)

if CUDA:
    model.cuda()

model.eval()

read_dir = time.time()

# Detection Phase
try:
    imlist = [os.path.join(os.path.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = [os.path.join(os.path.realpath('.'), images)]
except FileNotFoundError:
    print("No file or directory with name{}".format(images))
    exit()
    
if not os.path.exists(args.dest):
    os.mkdir(args.dest)
    
load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

#Pytorch Variables for images
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

#List contianing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

if CUDA:
    img_dim_list = im_dim_list.cuda()
    
leftover = 0
if (len(im_dim_list)%batch_size):
    leftover = 1
    
if batch_size != 1:
    num_batches = len(mlist)//batch_size + leftover
    im_batches = [torch.cat((im_batches[i*batch_size:min((i+1)*batch_size, len(im_batches))]))
                  for i in range(num_batches)]
    

write = 0
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    #load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)
    
    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thresh)
    
    end = time.time()
    
    if type(prediction)==int: # If no detection ws made
        for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} finished prediction in {1:6.3f} seconds".format(image.split("/")[-1], (end-start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("-------------------------------------------------------------------------")
        continue
        
    prediction[:,0] += i*batch_size
    
    if not write:
        output = prediction
        write = 1
        
    else:
        output = torch.cat((output, prediction))
        
    for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0])==im_id]
        print("{0:20s} finished prediction in {1:6.3f} seconds".format(image.split("/")[-1], (end-start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("-------------------------------------------------------------------------")
    if CUDA:
        torch.cuda.synchronize()
        
try:
    output
except NameError:
    print("No detection were made.")
    exit()

# Rescale output to match the dimension of the original image
im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
scaling_factor = torch.min(inp_dim/im_dim_list, 1)[0].view(-1,1)
output[:, [1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:, [2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

output[:,1:5] /= scaling_factor

# clip bounding box for those that extend outside the detected image
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i, 0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i, 0])
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", 'rb'))

draw = time.time()

def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    color = random.choice(colors)
    cls_ = int(x[-1])
    label = "{0}".format(classes[cls_])
    img = cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0]+3, c1[1]+t_size[1]+4
    img = cv2.rectangle(img, c1, c2, color, -1)
    img = cv2.putText(img, label, (c1[0], c1[1]+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img


result = list(map(lambda x: write(x, loaded_ims), output))

def out_path(x):
    x = os.path.normpath(x)
    temp = x.split(os.path.sep)
    s = f"{os.path.sep}".join(temp[:-2])
    return f"{os.path.sep}".join([s, args.dest, "det_"+os.path.basename(x)])

det_names = list(map(out_path, imlist))
list(map(cv2.imwrite, det_names, result))
end = time.time()

print("SUMMARY")
print("-------------------------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addreses", (load_batch-read_dir)))
print("{:25s}: {:2.3f}".format("Loading image batch", (start_det_loop-load_batch)))
print("{:25s}: {:2.3f}".format("Detection ("+str(len(imlist))+" images)", (output_recast-start_det_loop)))
print("{:25s}: {:2.3f}".format("Output Processing", (class_load-output_recast)))
print("{:25s}: {:2.3f}".format("Drawing Boxes", (end-draw)))
print("-------------------------------------------------------------------------")

torch.cuda.empty_cache()