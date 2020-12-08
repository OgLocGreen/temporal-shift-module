# include tf slim models repo (tf_models)
import os
import re
import pickle
import numpy as np
import sys
sys.path.append("tf_models/research/slim")
import tensorflow.compat.v1 as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.platform import gfile

import PIL

import tf_slim as slim
from nets import resnet_v1

EXPORT = False
IMPORT_PYTORCH = True
SPLIT_MODEL = False
SPLIT_EXPORT = True

RUN_MODEL = False and not SPLIT_MODEL
DUMP_QUANTIZE = True and not SPLIT_MODEL
QUANTIZE_KINETICS = True

# To generate images from videos and clips see README - Data Preparation
KINETICS_DIR = "" # TODO: Set to kinetics calibration data directory. (i.e. KINETICS_DIR/category/vid_id/img_*.jpg

IMAGENET_DIR = ""# Path to imagenet images for testing
QUANTIZE_VIDS = 12

torch_params = None

# Takes modulelist and returns parameter dict for linear layer
def torch_to_linear_state(module):
    init = tf.initializers.constant
    state = module.state_dict()

    linear_state = {
       "weights_initializer": init(state["weight"].numpy().T, verify_shape=True),
       "biases_initializer":  init(state["bias"].numpy(), verify_shape=True),
    }

    return linear_state

# Takes modulelist and returns parameter dict for Conv layer at index "i" and batchnorm at "i+1", and relu at "i+2"
def torch_to_conv_state(module, conv_idx, is_shift=False, is_downsample=False):
    """
    Translates pytorch conv2d, batchnorm, relu to tensorflow slim
    conv_idx: conv index within the module
    """

    # Passing to slim.conv2d using tf.initializers.constant
    # weights_initializer = "0.weight"
    # bias_intiializer defaults to zero already
    # normalizer_params = {
    # scale: True,  ## enable gamma
    # center: True, ## enable beta
    # epsilon: mod.eps
    # param_initializers:
    #     {
    #     moving_mean: 1.running_mean,
    #     moving_var: 1.running_var,
    #     beta: 1.weights,
    #     gamma: 1.bias
    #     }
    # }

    conv_str = f"conv{conv_idx}"
    bn_str = f"bn{conv_idx}"
    if is_downsample:
        conv_str = str(conv_idx)
        bn_str = str(conv_idx+1)

    if is_shift == 1:
        conv_str += ".net"

    init = tf.initializers.constant
    state = module.state_dict()

    def trans_w(x):
        if False:#depthwise:
            # Pytorch weights are [out_C, in_C/groups, filter_H, filter_W].
            # TF weights are [filter_H, filter_W, in_C, channel_mult].
            # Assume in_C = groups, so channel_mult = in_C/groups
            return np.moveaxis(x, [0,1,2,3], [2,3,0,1])
        else:
            # Pytorch weights are [out_C, in_C, filter_H, filter_W].
            # TF weights are [filter_H, filter_W, in_C, out_C].
            return np.moveaxis(x, [0,1,2,3], [3,2,0,1])


    conv2d_init = {
        ## Conv2D
        "weights_initializer": init(trans_w(state[conv_str+".weight"].numpy()), verify_shape=True),
        "biases_initializer": tf.zeros_initializer(),

        ## Batch Norm
        "normalizer_params": {
            "scale": True,
            "center": True,
            "epsilon": getattr(module, bn_str).eps,
            "param_initializers": {
                "moving_mean":       init(state[bn_str+".running_mean"].numpy(), verify_shape=True),
                "moving_variance":   init(state[bn_str+".running_var"].numpy(), verify_shape=True),
                "beta":              init(state[bn_str+".bias"].numpy(), verify_shape=True),
                "gamma":             init(state[bn_str+".weight"].numpy(), verify_shape=True),
            }
        }
    }

    return conv2d_init

def import_pytorch_weights():
    # Import TSM kinetics model
    import torch
    import torchvision
    from ops.models import TSN
    from ops.temporal_shift import TemporalShift
    global torch_params
    torch_params = {}
    
    torch_model = TSN(400, 8, 'RGB',
          base_model='resnet50',
          consensus_type='avg',
          img_feature_dim='256',
          pretrain='imagenet',
          is_shift=True, shift_div=8, shift_place='blockres',
          non_local=False,
          )

    if not os.path.exists("TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth"):  # checkpoint not downloaded
        print('Please download the checkpoint. See README.md - Kinetics400 - Uniform Sampling - TSM ResNet50 8*1clip.')
        return False

    ckpt = torch.load("TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth", map_location=torch.device('cpu'))['state_dict']
    state_dict = {'.'.join(k.split('.')[1:]): v for k, v in ckpt.items()}

    torch_model.load_state_dict(state_dict)

    i = 0
    resnet_children = torch_model.children()
    resnet = next(resnet_children)
    classifier = next(resnet_children)

    # Translate root convolutional block
    params = torch_to_conv_state(resnet, 1)
    torch_params["root"] = params

    params = torch_to_linear_state(classifier)
    torch_params["classifier"] = params

    for name,module in resnet.named_children():
        print(name, type(module))
        is_layer = isinstance(module, torch.nn.Sequential)

        if is_layer:
            block_name = "block" + re.search("layer(\d+)", name).group(1)
            torch_params[block_name] = {}

            for bottleneck_idx, bottleneck in module.named_children():
                if not isinstance(bottleneck, torchvision.models.resnet.Bottleneck):
                    continue
                bottleneck_name = f"bottleneck{bottleneck_idx}"
                torch_params[block_name][bottleneck_name] = {}

                for op_name, op in bottleneck.named_children():
                    is_conv  = isinstance(op, torch.nn.Conv2d)
                    is_shift = isinstance(op, TemporalShift)
                    is_downsample = isinstance(op, torch.nn.Sequential)
                    print(f"{block_name}:", op_name, op)
                    if is_conv or is_shift:
                        conv_idx = int(re.search("conv(\d)", op_name).group(1))
                        params = torch_to_conv_state(bottleneck, conv_idx, is_shift)
                        torch_params[block_name][bottleneck_name][op_name] = params
                    elif is_downsample:
                        print(op.state_dict().keys())
                        assert "shortcut" not in torch_params[block_name]
                        params = torch_to_conv_state(op, 0, False, True)
                        torch_params[block_name][bottleneck_name]["shortcut"] = params


        continue
        if is_conv or is_bn:
            layer_info = re.search('layer(\d)', name)
            conv_info = re.search('conv(\d)', name)
            bn_info = re.search('bn(\d)', name)
            down_info = re.search('downsample.(\d)', name)
            block_num = 0
            if layer_info:
                # Internal resnet blocks
                print("LAYER: ", layer_info.group(1))
                block_num = layer_info.group(1)

            if is_conv:
                if down_info:
                    print("DOWN_CONV: ", down_info.group(1))
                else:
                    print("CONV: ", conv_info.group(1))
            elif is_bn:
                if down_info:
                    print("DOWN_BN: ", down_info.group(1))
                else:
                    print("BN: ", bn_info.group(1))
        elif is_linear:
            pass




    if not os.path.exists("TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth"):
        return False

def export_quantize_info(out_path, inputs, output_node_names):
    with open(out_path, 'w') as f:
        in_shapes = []
        first = True
        f.write("--input_nodes \n")
        for i,op in enumerate(inputs):
            name = op.name.split(':')[0]
            if type(op) == tf.Operation:
                in_shapes.append(op.outputs[0].shape)
            else:
                in_shapes.append(op.shape)

            if not first:
                f.write(",")
            first = False
            f.write(name)
        f.write("\n\n")

        first = True
        f.write("--input_shapes \n")
        for i,shape in enumerate(in_shapes):
            if not first:
                f.write(":")
            first = False
            f.write(f"{shape[0]},{shape[1]},{shape[2]},{shape[3]}")
        f.write("\n\n")

        first = True
        f.write("--output_nodes \n")
        for i,out in enumerate(output_node_names):
            if not first:
                f.write(",")
            first = False
            f.write(out)

net = None
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    in_imgs = tf.placeholder(tf.float32, shape=(8,224,224,3), name='in_imgs')

    kernel_initializer = None
    bias_initializer = tf.zeros_initializer()
    if IMPORT_PYTORCH:
        import_pytorch_weights()
        kernel_initializer = torch_params["classifier"]["weights_initializer"]
        bias_initializer  = torch_params["classifier"]["biases_initializer"]
    print(torch_params["root"].keys())
    print(torch_params["classifier"].keys())
    print(torch_params["block1"].keys())
    print(torch_params["block1"]["bottleneck0"])

    net, endpoints = resnet_v1.resnet_v1_50(in_imgs,is_training=False, global_pool=False, output_stride=None, initializers=torch_params, split_model=SPLIT_MODEL, insert_shift=not SPLIT_MODEL)

    net = tf.nn.avg_pool(net, [1,7,7,1], 1, "VALID", name="AvgPool")
    net = tf.squeeze(net, (1,2))
    net = tf.layers.dense(net, 400, use_bias=True, trainable = False,
                       kernel_initializer = kernel_initializer,
                       bias_initializer  = bias_initializer,
                       name="Linear")

    # If exporting to Vitis, remove mean op from graph
    if DUMP_QUANTIZE or RUN_MODEL:
        net = tf.reduce_mean(net, axis=0)
    net = tf.identity(net, name="output")


output_node_names = []
split_outputs = {}
split_inputs = {}

with tf.Session() as sess:
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess.run(tf.global_variables_initializer())

    output_node_names.append("output")
    if SPLIT_MODEL:
        # Get previous unit in the model.
        # If first unit in the block, return last unit of previous block.
        # If first unit of first block, return root block (block0)
        resnet_block_lens = [3, 4, 6, 3]
        def get_prev_unit(block_idx, unit_idx):
            if unit_idx == 1:
                if block_idx == 1:
                    return (0,1)
                else:
                    return(block_idx-1, resnet_block_lens[block_idx-2])
            else:
                return (block_idx, unit_idx-1)

        for op in graph.get_operations():
            if "prev_conv_output" in op.name or "prev_shortcut_output" in op.name:
                block_num = int(re.search('block(\d)', op.name).group(1))
                unit_num = int(re.search('unit_(\d)', op.name).group(1))
                output_node_names.append(op.name)

                prev_block,prev_unit = get_prev_unit(block_num, unit_num)

                if prev_block not in split_outputs:
                    split_outputs[prev_block] = {}
                if prev_unit not in split_outputs[prev_block]:
                    split_outputs[prev_block][prev_unit] = {}
                if "prev_conv_output" in op.name:
                    split_outputs[prev_block][prev_unit]["conv_out"] = op
                else:
                    split_outputs[prev_block][prev_unit]["shortcut_out"] = op
            elif "conv_input" in op.name or "shortcut_input" in op.name:
                block_num = int(re.search('block(\d)', op.name).group(1))
                unit_num = int(re.search('unit_(\d)', op.name).group(1))

                if block_num not in split_inputs:
                    split_inputs[block_num] = {}
                if unit_num not in split_inputs[block_num]:
                    split_inputs[block_num][unit_num] = {}
                if "conv_input" in op.name:
                    split_inputs[block_num][unit_num]["conv_in"] = op
                else:
                    split_inputs[block_num][unit_num]["shortcut_in"] = op


        # Add outputs for the final unit of the final block
        split_outputs[len(resnet_block_lens)][resnet_block_lens[-1]] = {}
        split_outputs[len(resnet_block_lens)][resnet_block_lens[-1]]["conv_out"] = net

        # Add inputs for the root block
        split_inputs[0] = {}
        split_inputs[0][1] = {}
        split_inputs[0][1]["conv_in"] = in_imgs

    if RUN_MODEL or DUMP_QUANTIZE:
        assert not SPLIT_MODEL

        input_dir = KINETICS_DIR if QUANTIZE_KINETICS else IMAGENET_DIR
        assert os.path.isdir(input_dir)

        # Initialize with root block info (no shortcut input)
        inters = ["output:0", "in_imgs:0"]
        # Dictionaries that map [block_num][unit_num] -> inter index (inter index = output index)
        conv_in_dict = {0: {1: 1}}
        shortcut_in_dict = {0: {1: -1}}
        for op in graph.get_operations():
            block = re.search("block(\d)/", op.name)
            unit = re.search("unit_(\d)/", op.name)
            #conv_in_search = re.search("bottleneck_v1/prev_conv_output$", op.name)
            conv_in_search = re.search("bottleneck_v1/shifted_input$", op.name)
            shortcut_in_search = re.search("bottleneck_v1/prev_shortcut_output$", op.name)

            if not block:
                continue

            block = int(block.group(1))
            unit = int(unit.group(1))

            if block not in conv_in_dict:
                conv_in_dict[block] = {}
                shortcut_in_dict[block] = {}

            if conv_in_search:
                conv_in_dict[block][unit] = len(inters)
                inters.append(op.name + ":0")
            elif shortcut_in_search:
                shortcut_in_dict[block][unit] = len(inters)
                inters.append(op.name + ":0")

        img_paths = []
        if not QUANTIZE_KINETICS:
            dir_imgs = sorted(os.listdir(input_dir))
            for i in range(0, len(dir_imgs), 8):
                imgs = [os.path.join(input_dir,dir_imgs[j]) for j in range(i, i+8)]
                img_paths.append(imgs)
                break
        else:
            # Calibrate off of first category, first vid found for now
            cat_dirs = sorted(os.listdir(input_dir))
            cat_dirs = [os.path.join(input_dir, x) for x in cat_dirs]

            vids_per_cat = QUANTIZE_VIDS / len(cat_dirs)
           
            for cat_dir in cat_dirs:
                for vid_num, vid_dir in enumerate(sorted(os.listdir(cat_dir))):
                    if vid_num >= vids_per_cat:
                        break
                    vid_path = os.path.join(cat_dir, vid_dir)
                    dir_imgs = sorted([os.path.join(vid_path, path) for path in os.listdir(vid_path)])
                    vid_imgs = []
                    tick = len(dir_imgs) / 8.0
                    for seg_num in range(8):
                        i = int(tick / 2.0 + tick * seg_num)
                        vid_imgs.append(dir_imgs[i])

                    img_paths.append(vid_imgs)

        dump_data = {}
        for vid_num,p_imgs in enumerate(img_paths):
            print(f"Processing calib data # {vid_num}...", file=sys.stderr)
            imgs = []
            for i in range(8):
                img = PIL.Image.open(p_imgs[i])

                w,h = img.size
                new_w = 0
                new_h = 0
                if w > h:
                    new_h = 256
                    new_w = (256*w)//h
                else:
                    new_w = 256
                    new_h = (256*h)//w
                img = img.resize((new_w, new_h), PIL.Image.BILINEAR)
                left = (new_w - 224)//2
                top = (new_h - 224)//2
                img = img.crop((left, top, left+224, top+224))
                img = np.array(img)/255.0
                img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                imgs.append(img)

            inputs = np.array(imgs)
            outputs = sess.run(inters, {"in_imgs:0":inputs})
            print("MAX = ", outputs[0].argmax(), " = ", outputs[0].max())
            test = [[1,0], [5,0], [11,0]]
            for x in outputs[0]:
                for a in test:
                    if x >= outputs[0][a[0]]:
                        a[1] += 1

            for a in test:
                print(f"[{a[0]}] top {a[1]} >= {outputs[0][a[0]]}")

            if DUMP_QUANTIZE:
                for block_num,units in conv_in_dict.items():
                    for unit_num,conv_idx in units.items():
                        conv = outputs[conv_idx]
                        print(f"({block_num}, {unit_num}) = {conv.shape}")
                        shortcut = np.empty(shape=(0,0))
                        if block_num > 0:
                            shortcut = outputs[shortcut_in_dict[block_num][unit_num]]
                        if unit_num == 1:
                            dump_data[block_num] = {}

                        dump_data[block_num][unit_num] = {"conv_in": conv.tolist(),
                                                          "shortcut_in": shortcut.tolist()}

            if DUMP_QUANTIZE:
                print("Dumping Quantize Data...", file=sys.stderr)
                for block_num,units in conv_in_dict.items():
                    for unit_num,conv_idx in units.items():
                        dump_file = os.path.join("resnet50_tf_split_export", f"resnet50_tf_split_{block_num}_{unit_num}", f"inputs_{vid_num}.pickle")
                        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
                        with open(dump_file, 'wb') as f:
                            pickle.dump(dump_data[block_num][unit_num], f, pickle.HIGHEST_PROTOCOL)
                del dump_data
                dump_data = {}




    if EXPORT:
        print(f"Saving model...", file=sys.stderr)
        saver = tf.train.Saver()
        model_name = "resnet50_tf"
        if SPLIT_MODEL:
            model_name += "_split"

        save_dir = os.path.join(".", model_name)

        print(f"Saving model to {save_dir}...")
        ckpt_file = saver.save(sess, os.path.join(save_dir, model_name + ".ckpt"))
        pbtxt_file = model_name + ".pbtxt"
        tf.train.write_graph(graph_or_graph_def=input_graph_def, logdir=save_dir, name=pbtxt_file, as_text=True)

        pbtxt_path = os.path.join(save_dir, pbtxt_file)
        pb_path = os.path.join(save_dir, model_name + ".pb")
        frozen_graph_def = freeze_graph.freeze_graph(input_graph=pbtxt_path, input_saver='', input_binary=False, input_checkpoint=ckpt_file, output_node_names=",".join(output_node_names), output_graph=pb_path, restore_op_name="save/restore_all", filename_tensor_name="save/Const:0", clear_devices=True, initializer_nodes="")

print("DONE")

### Save a frozen graph for each disconnected portion
if EXPORT and SPLIT_MODEL and SPLIT_EXPORT:
    base_dir = "resnet50_tf_split_export"
    print(f"Exporting split graphs to {base_dir}...")
    for block_num,unit_dict in split_outputs.items():
        for unit_num,out_nodes in unit_dict.items():
            tf.reset_default_graph()

            output_node_names = [node.name.split(':')[0] for _,node in out_nodes.items()]
            input_nodes = [node for _,node in split_inputs[block_num][unit_num].items()]

            split_graph_def = tf.graph_util.extract_sub_graph(frozen_graph_def, output_node_names)
            with tf.Session() as split_sess:
                tf.graph_util.import_graph_def(split_graph_def, name="")
                split_graph = tf.get_default_graph()

            model_name = f"resnet50_tf_split_{block_num}_{unit_num}"
            save_dir = os.path.join(".", base_dir, model_name)
            pb_path = os.path.join(save_dir, model_name + ".pb")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with gfile.GFile(pb_path, "wb") as f:
                f.write(split_graph_def.SerializeToString())
            export_quantize_info(os.path.join(save_dir, "quantize_info.txt"), input_nodes, output_node_names)


