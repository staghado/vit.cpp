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

- [&#10004;] **Read the image**
  - [ ] Load the image from a file name
  - [ ] Divide it into patches
- [ ] **Create a ViT object**
  - [ ] Create a config to hold hparams
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
- [ ] **Convert the PyTorch weights**
  - [ ] Use ggml tensor format to load the params
  - [ ] Validate the weights
- [ ] **Test the inference**
  - [ ] Run inference on a sample image
  - [ ] Compare with PyTorch output
  - [ ] Benchmark inference speed vs. Pytorch models