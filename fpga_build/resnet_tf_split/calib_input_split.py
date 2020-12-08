import os
from PIL import Image
import numpy as np
import random
import pickle

IMAGENET_PATH = "/MEng/Data/ILSVRC2012_img_val/"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, .225]

CALIB_BASE_PATH=os.getenv("CALIB_BASE_PATH")
if CALIB_BASE_PATH is None:
    raise ValueError("Environment variable CALIB_BASE_PATH not set")

CALIB_BLOCK=os.getenv("CALIB_BLOCK")
if CALIB_BLOCK is None:
    raise ValueError("Environment variable CALIB_BLOCK not set")
CALIB_UNIT=os.getenv("CALIB_UNIT")
if CALIB_UNIT is None:
    raise ValueError("Environment variable CALIB_UNIT not set")


quantize_info_path = os.path.join(CALIB_BASE_PATH, f"resnet50_tf_split_{CALIB_BLOCK}_{CALIB_UNIT}/quantize_info.txt")
input_info_path = os.path.join(CALIB_BASE_PATH, f"resnet50_tf_split_{CALIB_BLOCK}_{CALIB_UNIT}/")

input_shapes = {}
with open(quantize_info_path) as f:
    lines = f.readlines()
    raw_input_names = []
    raw_input_shapes = []
    for i in range(len(lines)):
        if "--input_nodes" in lines[i]:
            raw_input_names = lines[i+1].rstrip()
        if "--input_shapes" in lines[i]:
            raw_input_shapes = lines[i+1].rstrip()

    raw_input_names = raw_input_names.split(",")
    raw_input_shapes = raw_input_shapes.split(":")
    raw_input_shapes = [[int(x) for x in shape.split(',')] for shape in raw_input_shapes]
    input_shapes = dict(zip(raw_input_names, raw_input_shapes))


input_data = None

def input_fn(iter):
    with open(os.path.join(input_info_path, f"inputs_{iter}.pickle"), 'rb') as f:
        input_data = pickle.load(f)

    inputs = {}
    for name,shape in input_shapes.items():
        if "conv_in" in name or "in_imgs" in name:
            inputs[name] = np.array(input_data["conv_in"])
        else:
            inputs[name] = np.array(input_data["shortcut_in"])

    return inputs
