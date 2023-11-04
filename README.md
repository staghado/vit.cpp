# vit.cpp
The objective of the project is to :

* Create a simple C/C++ inference engine for Vision Transformer(ViT) models with no dependencies

The implementation is destined to be lightweigt and self-contained to be able to run it on different platforms.
This is a work in progress.

This is inspired by the following projects:
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Vision Transformer architecture

![Vision Transfomer overview](assets/image.png)

# To-Do List

- [&#10004;] **Image preprocessing**
  - [ ] Load the image from a file name
  - [ ] Divide it into patches
- [&#10004;] **Convert the PyTorch weights**
  - [ ] Use ggml tensor format to load the params
  - [ ] Validate the weights
- [ ] **Create a ViT object**
  - [&#10004;] Create a config to hold hparams
  - [ ] Create a ViT class
    - [ ] Embedding
    - [ ] ViT Encoder
        - [ ] ViT Embeddings
            - [ ] Patch Embeddings
            - [ ] [CLS] token
            - [ ] Positional Encodings
        - [ ] Transformer Encoder
            - [ ] Layer Norm
            - [ ] Self Attention
            - [ ] MLP
        - [ ] Pooling : takes the first hidden state
    - [ ] Classifier
- [ ] **Test the inference**
  - [ ] Run inference on a sample image
  - [ ] Compare with PyTorch output
  - [ ] Benchmark inference speed vs. PyTorch models

<details>
<summary>ViT Base PyTorch weights</summary>

    cls_token                   : torch.Size([1, 1, 768])
    pos_embed                   : torch.Size([1, 785, 768])
    patch_embed.proj.weight     : torch.Size([768, 3, 8, 8])
    patch_embed.proj.bias       : torch.Size([768])
    blocks.0.norm1.weight       : torch.Size([768])
    blocks.0.norm1.bias         : torch.Size([768])
    blocks.0.attn.qkv.weight    : torch.Size([2304, 768])
    blocks.0.attn.qkv.bias      : torch.Size([2304])
    blocks.0.attn.proj.weight   : torch.Size([768, 768])
    blocks.0.attn.proj.bias     : torch.Size([768])
    blocks.0.norm2.weight       : torch.Size([768])
    blocks.0.norm2.bias         : torch.Size([768])
    blocks.0.mlp.fc1.weight     : torch.Size([3072, 768])
    blocks.0.mlp.fc1.bias       : torch.Size([3072])
    blocks.0.mlp.fc2.weight     : torch.Size([768, 3072])
    blocks.0.mlp.fc2.bias       : torch.Size([768])
    blocks.1.norm1.weight       : torch.Size([768])
    blocks.1.norm1.bias         : torch.Size([768])
    blocks.1.attn.qkv.weight    : torch.Size([2304, 768])
    blocks.1.attn.qkv.bias      : torch.Size([2304])
    blocks.1.attn.proj.weight   : torch.Size([768, 768])
    blocks.1.attn.proj.bias     : torch.Size([768])
    blocks.1.norm2.weight       : torch.Size([768])
    blocks.1.norm2.bias         : torch.Size([768])
    blocks.1.mlp.fc1.weight     : torch.Size([3072, 768])
    blocks.1.mlp.fc1.bias       : torch.Size([3072])
    blocks.1.mlp.fc2.weight     : torch.Size([768, 3072])
    blocks.1.mlp.fc2.bias       : torch.Size([768])
    blocks.2.norm1.weight       : torch.Size([768])
    blocks.2.norm1.bias         : torch.Size([768])
    blocks.2.attn.qkv.weight    : torch.Size([2304, 768])
    blocks.2.attn.qkv.bias      : torch.Size([2304])
    blocks.2.attn.proj.weight   : torch.Size([768, 768])
    blocks.2.attn.proj.bias     : torch.Size([768])
    blocks.2.norm2.weight       : torch.Size([768])
    blocks.2.norm2.bias         : torch.Size([768])
    blocks.2.mlp.fc1.weight     : torch.Size([3072, 768])
    blocks.2.mlp.fc1.bias       : torch.Size([3072])
    blocks.2.mlp.fc2.weight     : torch.Size([768, 3072])
    blocks.2.mlp.fc2.bias       : torch.Size([768])
    blocks.3.norm1.weight       : torch.Size([768])
    blocks.3.norm1.bias         : torch.Size([768])
    blocks.3.attn.qkv.weight    : torch.Size([2304, 768])
    blocks.3.attn.qkv.bias      : torch.Size([2304])
    blocks.3.attn.proj.weight   : torch.Size([768, 768])
    blocks.3.attn.proj.bias     : torch.Size([768])
    blocks.3.norm2.weight       : torch.Size([768])
    blocks.3.norm2.bias         : torch.Size([768])
    blocks.3.mlp.fc1.weight     : torch.Size([3072, 768])
    blocks.3.mlp.fc1.bias       : torch.Size([3072])
    blocks.3.mlp.fc2.weight     : torch.Size([768, 3072])
    blocks.3.mlp.fc2.bias       : torch.Size([768])
    blocks.4.norm1.weight       : torch.Size([768])
    blocks.4.norm1.bias         : torch.Size([768])
    blocks.4.attn.qkv.weight    : torch.Size([2304, 768])
    blocks.4.attn.qkv.bias      : torch.Size([2304])
    blocks.4.attn.proj.weight   : torch.Size([768, 768])
    blocks.4.attn.proj.bias     : torch.Size([768])
    blocks.4.norm2.weight       : torch.Size([768])
    blocks.4.norm2.bias         : torch.Size([768])
    blocks.4.mlp.fc1.weight     : torch.Size([3072, 768])
    blocks.4.mlp.fc1.bias       : torch.Size([3072])
    blocks.4.mlp.fc2.weight     : torch.Size([768, 3072])
    blocks.4.mlp.fc2.bias       : torch.Size([768])
    blocks.5.norm1.weight       : torch.Size([768])
    blocks.5.norm1.bias         : torch.Size([768])
    blocks.5.attn.qkv.weight    : torch.Size([2304, 768])
    blocks.5.attn.qkv.bias      : torch.Size([2304])
    blocks.5.attn.proj.weight   : torch.Size([768, 768])
    blocks.5.attn.proj.bias     : torch.Size([768])
    blocks.5.norm2.weight       : torch.Size([768])
    blocks.5.norm2.bias         : torch.Size([768])
    blocks.5.mlp.fc1.weight     : torch.Size([3072, 768])
    blocks.5.mlp.fc1.bias       : torch.Size([3072])
    blocks.5.mlp.fc2.weight     : torch.Size([768, 3072])
    blocks.5.mlp.fc2.bias       : torch.Size([768])
    blocks.6.norm1.weight       : torch.Size([768])
    blocks.6.norm1.bias         : torch.Size([768])
    blocks.6.attn.qkv.weight    : torch.Size([2304, 768])
    blocks.6.attn.qkv.bias      : torch.Size([2304])
    blocks.6.attn.proj.weight   : torch.Size([768, 768])
    blocks.6.attn.proj.bias     : torch.Size([768])
    blocks.6.norm2.weight       : torch.Size([768])
    blocks.6.norm2.bias         : torch.Size([768])
    blocks.6.mlp.fc1.weight     : torch.Size([3072, 768])
    blocks.6.mlp.fc1.bias       : torch.Size([3072])
    blocks.6.mlp.fc2.weight     : torch.Size([768, 3072])
    blocks.6.mlp.fc2.bias       : torch.Size([768])
    blocks.7.norm1.weight       : torch.Size([768])
    blocks.7.norm1.bias         : torch.Size([768])
    blocks.7.attn.qkv.weight    : torch.Size([2304, 768])
    blocks.7.attn.qkv.bias      : torch.Size([2304])
    blocks.7.attn.proj.weight   : torch.Size([768, 768])
    blocks.7.attn.proj.bias     : torch.Size([768])
    blocks.7.norm2.weight       : torch.Size([768])
    blocks.7.norm2.bias         : torch.Size([768])
    blocks.7.mlp.fc1.weight     : torch.Size([3072, 768])
    blocks.7.mlp.fc1.bias       : torch.Size([3072])
    blocks.7.mlp.fc2.weight     : torch.Size([768, 3072])
    blocks.7.mlp.fc2.bias       : torch.Size([768])
    blocks.8.norm1.weight       : torch.Size([768])
    blocks.8.norm1.bias         : torch.Size([768])
    blocks.8.attn.qkv.weight    : torch.Size([2304, 768])
    blocks.8.attn.qkv.bias      : torch.Size([2304])
    blocks.8.attn.proj.weight   : torch.Size([768, 768])
    blocks.8.attn.proj.bias     : torch.Size([768])
    blocks.8.norm2.weight       : torch.Size([768])
    blocks.8.norm2.bias         : torch.Size([768])
    blocks.8.mlp.fc1.weight     : torch.Size([3072, 768])
    blocks.8.mlp.fc1.bias       : torch.Size([3072])
    blocks.8.mlp.fc2.weight     : torch.Size([768, 3072])
    blocks.8.mlp.fc2.bias       : torch.Size([768])
    blocks.9.norm1.weight       : torch.Size([768])
    blocks.9.norm1.bias         : torch.Size([768])
    blocks.9.attn.qkv.weight    : torch.Size([2304, 768])
    blocks.9.attn.qkv.bias      : torch.Size([2304])
    blocks.9.attn.proj.weight   : torch.Size([768, 768])
    blocks.9.attn.proj.bias     : torch.Size([768])
    blocks.9.norm2.weight       : torch.Size([768])
    blocks.9.norm2.bias         : torch.Size([768])
    blocks.9.mlp.fc1.weight     : torch.Size([3072, 768])
    blocks.9.mlp.fc1.bias       : torch.Size([3072])
    blocks.9.mlp.fc2.weight     : torch.Size([768, 3072])
    blocks.9.mlp.fc2.bias       : torch.Size([768])
    blocks.10.norm1.weight      : torch.Size([768])
    blocks.10.norm1.bias        : torch.Size([768])
    blocks.10.attn.qkv.weight   : torch.Size([2304, 768])
    blocks.10.attn.qkv.bias     : torch.Size([2304])
    blocks.10.attn.proj.weight  : torch.Size([768, 768])
    blocks.10.attn.proj.bias    : torch.Size([768])
    blocks.10.norm2.weight      : torch.Size([768])
    blocks.10.norm2.bias        : torch.Size([768])
    blocks.10.mlp.fc1.weight    : torch.Size([3072, 768])
    blocks.10.mlp.fc1.bias      : torch.Size([3072])
    blocks.10.mlp.fc2.weight    : torch.Size([768, 3072])
    blocks.10.mlp.fc2.bias      : torch.Size([768])
    blocks.11.norm1.weight      : torch.Size([768])
    blocks.11.norm1.bias        : torch.Size([768])
    blocks.11.attn.qkv.weight   : torch.Size([2304, 768])
    blocks.11.attn.qkv.bias     : torch.Size([2304])
    blocks.11.attn.proj.weight  : torch.Size([768, 768])
    blocks.11.attn.proj.bias    : torch.Size([768])
    blocks.11.norm2.weight      : torch.Size([768])
    blocks.11.norm2.bias        : torch.Size([768])
    blocks.11.mlp.fc1.weight    : torch.Size([3072, 768])
    blocks.11.mlp.fc1.bias      : torch.Size([3072])
    blocks.11.mlp.fc2.weight    : torch.Size([768, 3072])
    blocks.11.mlp.fc2.bias      : torch.Size([768])
    norm.weight                 : torch.Size([768])
    norm.bias                   : torch.Size([768])
    head.weight                 : torch.Size([1000, 768])
    head.bias                   : torch.Size([1000])
</details>