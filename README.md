# vit.cpp
The objective of the project is to :

* Create a simple C/C++ inference engine for Vision Transformer(ViT) models with no dependencies

The implementation is destined to be lightweight and self-contained to be able to run it on different platforms.

This is inspired by the following projects:
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)

## [This is a work in progress]
## Vision Transformer architecture

![Vision Transfomer overview](assets/image.png)

# Convert PyTorch to ggml format
    pip install torch timm
    python save_pth.py
    python convert-pth-to-ggml.py vit_base.pth . 1

# Build

    # build ggml and vit 
    mkdir build && cd build
    cmake .. && make -j4

    # or simply run after setting the rights (chmod +x if needed)
    ./build.sh

    # run inference
    ./bin/vit -t 8 -i ../assets/tench.jpg -m ../ggml-model-f16.bin

    # add per device optimizations
    # using OpenBLAS
    # using OpenMP

# Run

    usage: ./bin/vit [options]

    options:
      -h, --help            show this help message and exit
      -s SEED, --seed SEED  RNG seed (default: -1)
      -t N, --threads N     number of threads to use during computation (default: 4)
      -m FNAME, --model FNAME
                            model path (default: ../ggml-model-f16.bin)
      -i FNAME, --inp FNAME
                            input file (default: ../assets/tench.jpg)
      -e FLOAT, --epsilon
                            epsilon (default: 0.000001)


# To-Do List

- [&#10004;] **Image preprocessing**
  - [&#10004;] Load the image from a file name
  - [&#10004;] Create image patches

- [&#10004;] **Convert the PyTorch weights**
  - [&#10004;] Use ggml tensor format to load the params
  - [&#10004;] Validate the weights

- [ ] **Create a ViT object**
  - [&#10004;] Create a config to hold hparams
  - [&#10004;] Create a ViT struct
    - [&#10004;] ViT Encoder
        - [&#10004;] ViT Embeddings
            - [&#10004;] Patch Embeddings
            - [&#10004;] [CLS] token
            - [&#10004;] Positional Encodings
        - [&#10004;] Transformer Encoder
            - [&#10004;] Layer Norm
            - [&#10004;] Self Attention
            - [&#10004;] MLP
        - [&#10004;] Pooling
    - [&#10004;] Classifier

- [ ] **Add quantization**
  - [ ] 8-bit
  - [ ] 4-bit?

- [ ] **Test the inference**
  - [&#10004;] Run inference on a sample image
  - [&#10004;] Compare with PyTorch output
  - [ ] Benchmark inference speed vs. PyTorch models

<details>
<summary>ViT Base PyTorch weights</summary>

    cls_token                   : [1, 1, 768]
    pos_embed                   : [1, 785, 768]
    patch_embed.proj.weight     : [768, 3, 8, 8]
    patch_embed.proj.bias       : [768]
    blocks.0.norm1.weight       : [768]
    blocks.0.norm1.bias         : [768]
    blocks.0.attn.qkv.weight    : [2304, 768]
    blocks.0.attn.qkv.bias      : [2304]
    blocks.0.attn.proj.weight   : [768, 768]
    blocks.0.attn.proj.bias     : [768]
    blocks.0.norm2.weight       : [768]
    blocks.0.norm2.bias         : [768]
    blocks.0.mlp.fc1.weight     : [3072, 768]
    blocks.0.mlp.fc1.bias       : [3072]
    blocks.0.mlp.fc2.weight     : [768, 3072]
    blocks.0.mlp.fc2.bias       : [768]
    blocks.1.norm1.weight       : [768]
    blocks.1.norm1.bias         : [768]
    blocks.1.attn.qkv.weight    : [2304, 768]
    blocks.1.attn.qkv.bias      : [2304]
    blocks.1.attn.proj.weight   : [768, 768]
    blocks.1.attn.proj.bias     : [768]
    blocks.1.norm2.weight       : [768]
    blocks.1.norm2.bias         : [768]
    blocks.1.mlp.fc1.weight     : [3072, 768]
    blocks.1.mlp.fc1.bias       : [3072]
    blocks.1.mlp.fc2.weight     : [768, 3072]
    blocks.1.mlp.fc2.bias       : [768]
    blocks.2.norm1.weight       : [768]
    blocks.2.norm1.bias         : [768]
    blocks.2.attn.qkv.weight    : [2304, 768]
    blocks.2.attn.qkv.bias      : [2304]
    blocks.2.attn.proj.weight   : [768, 768]
    blocks.2.attn.proj.bias     : [768]
    blocks.2.norm2.weight       : [768]
    blocks.2.norm2.bias         : [768]
    blocks.2.mlp.fc1.weight     : [3072, 768]
    blocks.2.mlp.fc1.bias       : [3072]
    blocks.2.mlp.fc2.weight     : [768, 3072]
    blocks.2.mlp.fc2.bias       : [768]
    blocks.3.norm1.weight       : [768]
    blocks.3.norm1.bias         : [768]
    blocks.3.attn.qkv.weight    : [2304, 768]
    blocks.3.attn.qkv.bias      : [2304]
    blocks.3.attn.proj.weight   : [768, 768]
    blocks.3.attn.proj.bias     : [768]
    blocks.3.norm2.weight       : [768]
    blocks.3.norm2.bias         : [768]
    blocks.3.mlp.fc1.weight     : [3072, 768]
    blocks.3.mlp.fc1.bias       : [3072]
    blocks.3.mlp.fc2.weight     : [768, 3072]
    blocks.3.mlp.fc2.bias       : [768]
    blocks.4.norm1.weight       : [768]
    blocks.4.norm1.bias         : [768]
    blocks.4.attn.qkv.weight    : [2304, 768]
    blocks.4.attn.qkv.bias      : [2304]
    blocks.4.attn.proj.weight   : [768, 768]
    blocks.4.attn.proj.bias     : [768]
    blocks.4.norm2.weight       : [768]
    blocks.4.norm2.bias         : [768]
    blocks.4.mlp.fc1.weight     : [3072, 768]
    blocks.4.mlp.fc1.bias       : [3072]
    blocks.4.mlp.fc2.weight     : [768, 3072]
    blocks.4.mlp.fc2.bias       : [768]
    blocks.5.norm1.weight       : [768]
    blocks.5.norm1.bias         : [768]
    blocks.5.attn.qkv.weight    : [2304, 768]
    blocks.5.attn.qkv.bias      : [2304]
    blocks.5.attn.proj.weight   : [768, 768]
    blocks.5.attn.proj.bias     : [768]
    blocks.5.norm2.weight       : [768]
    blocks.5.norm2.bias         : [768]
    blocks.5.mlp.fc1.weight     : [3072, 768]
    blocks.5.mlp.fc1.bias       : [3072]
    blocks.5.mlp.fc2.weight     : [768, 3072]
    blocks.5.mlp.fc2.bias       : [768]
    blocks.6.norm1.weight       : [768]
    blocks.6.norm1.bias         : [768]
    blocks.6.attn.qkv.weight    : [2304, 768]
    blocks.6.attn.qkv.bias      : [2304]
    blocks.6.attn.proj.weight   : [768, 768]
    blocks.6.attn.proj.bias     : [768]
    blocks.6.norm2.weight       : [768]
    blocks.6.norm2.bias         : [768]
    blocks.6.mlp.fc1.weight     : [3072, 768]
    blocks.6.mlp.fc1.bias       : [3072]
    blocks.6.mlp.fc2.weight     : [768, 3072]
    blocks.6.mlp.fc2.bias       : [768]
    blocks.7.norm1.weight       : [768]
    blocks.7.norm1.bias         : [768]
    blocks.7.attn.qkv.weight    : [2304, 768]
    blocks.7.attn.qkv.bias      : [2304]
    blocks.7.attn.proj.weight   : [768, 768]
    blocks.7.attn.proj.bias     : [768]
    blocks.7.norm2.weight       : [768]
    blocks.7.norm2.bias         : [768]
    blocks.7.mlp.fc1.weight     : [3072, 768]
    blocks.7.mlp.fc1.bias       : [3072]
    blocks.7.mlp.fc2.weight     : [768, 3072]
    blocks.7.mlp.fc2.bias       : [768]
    blocks.8.norm1.weight       : [768]
    blocks.8.norm1.bias         : [768]
    blocks.8.attn.qkv.weight    : [2304, 768]
    blocks.8.attn.qkv.bias      : [2304]
    blocks.8.attn.proj.weight   : [768, 768]
    blocks.8.attn.proj.bias     : [768]
    blocks.8.norm2.weight       : [768]
    blocks.8.norm2.bias         : [768]
    blocks.8.mlp.fc1.weight     : [3072, 768]
    blocks.8.mlp.fc1.bias       : [3072]
    blocks.8.mlp.fc2.weight     : [768, 3072]
    blocks.8.mlp.fc2.bias       : [768]
    blocks.9.norm1.weight       : [768]
    blocks.9.norm1.bias         : [768]
    blocks.9.attn.qkv.weight    : [2304, 768]
    blocks.9.attn.qkv.bias      : [2304]
    blocks.9.attn.proj.weight   : [768, 768]
    blocks.9.attn.proj.bias     : [768]
    blocks.9.norm2.weight       : [768]
    blocks.9.norm2.bias         : [768]
    blocks.9.mlp.fc1.weight     : [3072, 768]
    blocks.9.mlp.fc1.bias       : [3072]
    blocks.9.mlp.fc2.weight     : [768, 3072]
    blocks.9.mlp.fc2.bias       : [768]
    blocks.10.norm1.weight      : [768]
    blocks.10.norm1.bias        : [768]
    blocks.10.attn.qkv.weight   : [2304, 768]
    blocks.10.attn.qkv.bias     : [2304]
    blocks.10.attn.proj.weight  : [768, 768]
    blocks.10.attn.proj.bias    : [768]
    blocks.10.norm2.weight      : [768]
    blocks.10.norm2.bias        : [768]
    blocks.10.mlp.fc1.weight    : [3072, 768]
    blocks.10.mlp.fc1.bias      : [3072]
    blocks.10.mlp.fc2.weight    : [768, 3072]
    blocks.10.mlp.fc2.bias      : [768]
    blocks.11.norm1.weight      : [768]
    blocks.11.norm1.bias        : [768]
    blocks.11.attn.qkv.weight   : [2304, 768]
    blocks.11.attn.qkv.bias     : [2304]
    blocks.11.attn.proj.weight  : [768, 768]
    blocks.11.attn.proj.bias    : [768]
    blocks.11.norm2.weight      : [768]
    blocks.11.norm2.bias        : [768]
    blocks.11.mlp.fc1.weight    : [3072, 768]
    blocks.11.mlp.fc1.bias      : [3072]
    blocks.11.mlp.fc2.weight    : [768, 3072]
    blocks.11.mlp.fc2.bias      : [768]
    norm.weight                 : [768]
    norm.bias                   : [768]
    head.weight                 : [1000, 768]
    head.bias                   : [1000]
</details>