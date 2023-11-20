"""
This script converts the PyTorch weights of a Vision Transformer to the ggml file format.

It accepts a timm model name and returns the converted weights in the same directory as the script.

You can also specify the float type : 0 for float32, 1 for float16

usage: convert-pth-to-ggml.py [-h] model_name {0,1}

positional arguments:
  model_name  timm model name
  {0,1}       float type: 0 for float32, 1 for float16

optional arguments:
  -h, --help  show this help message and exit

"""

import argparse
import timm
from timm.data import ImageNetInfo, infer_imagenet_subset
import struct
import numpy as np


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert PyTorch weights of a Vision Transformer to the ggml file format.')
    parser.add_argument('model_name', type=str, default='vit_base_patch8_224.augreg2_in21k_ft_in1k', help='timm model name')
    parser.add_argument('ftype', type=int, choices=[0, 1], default=1, help='float type: 0 for float32, 1 for float16')
    args = parser.parse_args()

    # Output file name
    fname_out = f"./ggml-model-{['f32', 'f16'][args.ftype]}.bin"

    # Load the pretrained model
    timm_model = timm.create_model(args.model_name, pretrained=True)

    # Create id2label dictionary
    imagenet_subset = infer_imagenet_subset(timm_model)
    dataset_info = ImageNetInfo(imagenet_subset)
    id2label = {i: dataset_info.index_to_description(i) for i in range(dataset_info.num_classes())}

    # Hyperparameters
    hparams = {
        "hidden_size": timm_model.embed_dim,
        "num_hidden_layers": len(timm_model.blocks),
        "num_attention_heads": timm_model.blocks[0].attn.num_heads,
        "num_classes": timm_model.num_classes,
        "patch_size": timm_model.patch_embed.patch_size[0],
        "img_size": timm_model.patch_embed.img_size[0]
    }

    # Write to file
    with open(fname_out, "wb") as fout:
        fout.write(struct.pack("i", 0x67676d6c))  # Magic: ggml in hex
        for param in hparams.values():
            fout.write(struct.pack("i", param))
        fout.write(struct.pack("i", args.ftype))

        # Write id2label dictionary to the file
        write_id2label(fout, id2label)

        # Process and write model weights
        for k, v in timm_model.state_dict().items():
            print("Processing variable: " + k + " with shape: ", v.shape, " and type: ", v.dtype)
            process_and_write_variable(fout, k, v, args.ftype)
        
        print("Done. Output file: " + fname_out)

def write_id2label(file, id2label):
    file.write(struct.pack("i", len(id2label)))
    for key, value in id2label.items():
        file.write(struct.pack("i", key))
        encoded_value = value.encode('utf-8')
        file.write(struct.pack("i", len(encoded_value)))
        file.write(encoded_value)

def process_and_write_variable(file, name, tensor, ftype):
    data = tensor.numpy()
    ftype_cur = 1 if ftype == 1 and tensor.ndim != 1 and name not in ["pos_embed", "cls_token"] else 0
    data = data.astype(np.float32) if ftype_cur == 0 else data.astype(np.float16)

    if name == "patch_embed.proj.bias":
        data = data.reshape(1, data.shape[0], 1, 1)

    str_name = name.encode('utf-8')
    file.write(struct.pack("iii", len(data.shape), len(str_name), ftype_cur))
    for dim_size in reversed(data.shape):
        file.write(struct.pack("i", dim_size))
    file.write(str_name)
    data.tofile(file)

if __name__ == "__main__":
    main()