"""
This script converts the PyTorch weights of a Vision Transformer to the ggml file format.
"""

import argparse
import struct
import string
import re

import numpy as np
from collections import OrderedDict

import torch 
import torch.nn as nn
import logging
import torch.utils.model_zoo as model_zoo

import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models import create_model

_logger = logging.getLogger(__name__)

__all__ = [
    'vitstr_tiny_patch16_224', 
    'vitstr_small_patch16_224', 
    'vitstr_base_patch16_224',
]


def create_vitstr(num_tokens, model=None, checkpoint_path=''):
    vitstr = create_model(
        model,
        pretrained=True,
        num_classes=num_tokens,
        checkpoint_path=checkpoint_path)

    # might need to run to get zero init head for transfer learning
    vitstr.reset_classifier(num_classes=num_tokens)

    return vitstr

class ViTSTR(VisionTransformer):
    '''
    ViTSTR is basically a ViT that uses DeiT weights.
    Modified head to support a sequence of characters prediction for STR.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x, seqlen: int =25):
        x = self.forward_features(x)
        x = x[:, :seqlen]

        # batch, seqlen, embsize
        b, s, e = x.size()
        x = x.reshape(b*s, e)
        x = self.head(x).view(b, s, self.num_classes)
        return x


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=1, filter_fn=None, strict=True):
    '''
    Loads a pretrained checkpoint
    From an older version of timm
    '''
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(cfg['url'], progress=True, map_location='cpu')
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        key = conv1_name + '.weight'
        if key in state_dict.keys():
            _logger.info('(%s) key found in state_dict' % key)
            conv1_weight = state_dict[conv1_name + '.weight']
        else:
            _logger.info('(%s) key NOT found in state_dict' % key)
            return
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    print("Loading pre-trained vision transformer weights from %s ..." % cfg['url'])
    model.load_state_dict(state_dict, strict=strict)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@register_model
def vitstr_tiny_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = ViTSTR(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, **kwargs)

    model.default_cfg = _cfg(
            #url='https://github.com/roatienza/public/releases/download/v0.1-deit-tiny/deit_tiny_patch16_224-a1311bcf.pth'
            url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'
    )

    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 1), filter_fn=_conv_filter)
    return model

@register_model
def vitstr_small_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = ViTSTR(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            #url="https://github.com/roatienza/public/releases/download/v0.1-deit-small/deit_small_patch16_224-cd65a155.pth"
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 1), filter_fn=_conv_filter)
    return model

@register_model
def vitstr_base_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = ViTSTR(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            #url='https://github.com/roatienza/public/releases/download/v0.1-deit-base/deit_base_patch16_224-b5f2ef4d.pth'
            url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 1), filter_fn=_conv_filter)
    return model


GGML_MAGIC = 0x67676d6c


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Convert PyTorch weights of a Vision Transformer to the ggml file format."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vitstr_tiny_patch16_224_aug.pth",
        help="model name",
    )
    parser.add_argument(
        "--ftype",
        type=int,
        choices=[0, 1],
        default=1,
        help="float type: 0 for float32, 1 for float16",
    )
    args = parser.parse_args()

    # Output file name
    fname_out = f"./ggml-model-{['f32', 'f16'][args.ftype]}.gguf"

    # load the chekcpoint and rename
    state_dict = torch.load(args.model_name, map_location=torch.device('cpu'))

    # create a new OrderedDict with modified keys
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.vitstr.", "")
        new_state_dict[new_key] = value

    # save to disk
    torch.save(new_state_dict, f"renamed_{args.model_name}")

    # define vocab 
    vocab = ['[GO]', '[s]'] + list(string.printable[:-6])
    num_tokens = len(vocab)

    # load the pretrained model
    timm_model = create_vitstr(num_tokens=num_tokens,
                                model=re.sub(r'vitstr_(.*?)_patch16_224.*?\.pth', r'vitstr_\1_patch16_224', args.model_name), 
                                checkpoint_path="")
    
    # load the pre-trained model into the initialized model
    timm_model.load_state_dict(new_state_dict);

    # label names
    id2label = {i: c for i, c in enumerate(vocab)}
    print(id2label)

    # Hyperparameters
    hparams = {
        "hidden_size": timm_model.embed_dim,
        "num_hidden_layers": len(timm_model.blocks),
        "num_attention_heads": timm_model.blocks[0].attn.num_heads,
        "num_classes": timm_model.num_classes,
        "patch_size": timm_model.patch_embed.patch_size[0],
        "img_size": timm_model.patch_embed.img_size[0],
    }

    print(hparams)

    # Write to file
    with open(fname_out, "wb") as fout:
        fout.write(struct.pack("i", GGML_MAGIC))  # Magic: ggml in hex
        for param in hparams.values():
            fout.write(struct.pack("i", param))
        fout.write(struct.pack("i", args.ftype))

        # Write id2label dictionary to the file
        write_id2label(fout, id2label)

        # Process and write model weights
        for k, v in timm_model.state_dict().items():
            if k.startswith("norm_pre"):
                print(k)
                continue
            print(
                "Processing variable: " + k + " with shape: ",
                v.shape,
                " and type: ",
                v.dtype,
            )
            process_and_write_variable(fout, k, v, args.ftype)

        print("Done. Output file: " + fname_out)


def write_id2label(file, id2label):
    file.write(struct.pack("i", len(id2label)))
    for key, value in id2label.items():
        file.write(struct.pack("i", key))
        encoded_value = value.encode("utf-8")
        file.write(struct.pack("i", len(encoded_value)))
        file.write(encoded_value)


def process_and_write_variable(file, name, tensor, ftype):
    data = tensor.numpy()
    ftype_cur = (
        1
        if ftype == 1 and tensor.ndim != 1 and name not in ["pos_embed", "cls_token"]
        else 0
    )
    data = data.astype(np.float32) if ftype_cur == 0 else data.astype(np.float16)

    if name == "patch_embed.proj.bias":
        data = data.reshape(1, data.shape[0], 1, 1)

    str_name = name.encode("utf-8")
    file.write(struct.pack("iii", len(data.shape), len(str_name), ftype_cur))
    for dim_size in reversed(data.shape):
        file.write(struct.pack("i", dim_size))
    file.write(str_name)
    data.tofile(file)


if __name__ == "__main__":
    main()