# vit.cpp
The objective of the project is to create a C++ inference engine for Vision Transformer(ViT) models 
using [ggml](https://github.com/ggerganov/ggml) which focuses on performance on edge devices.

The implementation is destined to be lightweight and self-contained to be able to run it on different platforms.

Per device optimizations are possible and quantization techniques will be added soon.

### [This is a work in progress]

## Vision Transformer architecture

The implemented architecture is based on the original Vision Transformer from:
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

<p align="center">
  <img src="assets/image.png" alt="Vision Transformer overview" width="60%" height="auto">
</p>
<p align="center">
  ViT architecture. Taken from the <a href="https://arxiv.org/abs/2010.11929">original paper</a>.
</p>


## Convert PyTorch to GGUF

    # clone the repo recursively
    git clone --recurse-submodules https://github.com/staghado/vit.cpp.git

    cd vit.cpp

    # install torch and timm
    pip install torch timm

    # list available models if needed, note that not all models are supported
    python convert-pth-to-ggml.py --list

    # convert the weights to gguf : vit tiny with patch size of 16 and an image 
    # size of 384 pre-trained on ImageNet21k and fine-tuned on ImageNet1k
    python convert-pth-to-ggml.py --model_name vit_tiny_patch16_384.augreg_in21k_ft_in1k --ftype 1

## Build
### Simple build
    # build ggml and vit 
    mkdir build && cd build
    cmake .. && make -j4

    # run inference
    ./bin/vit -t 4 -m ../ggml-model-f16.gguf -i ../assets/tench.jpg

The optimal number of threads to use depends on many factors and more is not always better. Usually using a number of threads equal to the number of available physical cores gives the best performance in terms of speed.

### Per device optimizations

Generate per-device instructions that work best for the given machine rather than using general CPU instructions.
This can be done by specifying `-march=native` in the compiler flags.
  * Multi-threading and vectorization
  * Loop transformations(unrolling)

#### For AMD host processors

You can use a specialized compiler released by AMD to make full use of your specific processor's architecture.
Read more here : [AMD Optimizing C/C++ and Fortran Compilers (AOCC)](https://www.amd.com/en/developer/aocc.html)

You can follow the given instructions to install the AOCC compiler.

Note : For my AMD Ryzen™ 7 3700U, the improvements were not very significant but for more recent processors there could be a gain in using a specialized compiler.

### Using OpenMP

Additionally compile with OpenMP by specifying the `-fopenmp` flag to the compiler in the CMakeLists file,
allowing multithreaded runs. Make sure to also enable multiple threads when running, e.g.:

    OMP_NUM_THREADS=4 ./bin/vit -t 4 -m ../ggml-model-f16.bin -i ../assets/tench.jpg

## Run

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

## Benchmark against PyTorch

First experiments on Apple M1 show inference speedups(up to 6x faster for base model) compared to native PyTorch inference. 
Extensive experiments will be conducted to verify this.
A comparison with ONNX models will be added as well.

### ViT inference

You can efficiently run ViT inference on the CPU.
Memory requirements and inference speed on AMD Ryzen 7 3700U(4 cores, 8 threads) for both native PyTorch and `vit.cpp`. 
Using 4 threads gives better results for my machine. The reported results of inference speed correspond to 10 runs averages for both PyTorch and `vit.cpp`.

| Model  | Mem(PyTorch)  | Mem            | Speed(PyTorch) | Speed          |
| :----: | :-----------: | :------------: | :------------: | :------------: |
| tiny   | ~780 MB       | **~20 MB**     | 431 ms         | **120 ms**     |
| small  | ~965 MB       | **~52 MB**     | 780 ms         | **463 ms**     |
| base   | ~1609 MB      | **~179 MB**    | 2393 ms        | **1441 ms**    |
| large  | ~3865 MB      | **~597 MB**    | 8151 ms        | **4892 ms**    |

> **Note:** The models used are of the form `vit_{size}_patch16_224.augreg_in21k_ft_in1k`.

### Benchmark on your machine

In order to test the inference speed on your machine, you can run the following scripts:

    chmod +x scripts/benchmark.*

    # run the benchmark of PyTorch
    # install memory_profiler first
    pip install memory_profiler
    python scripts/benchmark.py

    # run the benchmark of vit.cpp
    ./scripts/benchmark.sh

Both scripts use 4 threads by default. In Python, the `threadpoolctl` library is used to limit the number of threads used by PyTorch.

## To-Do List
- [ ] **Implement Bicubic Interpolation**: 

  For now the image resizing is done with bilinear interpolation but the models were trained with bicubic interpolation, this could result in the loss of performance.

- [ ] **Add quantization**
  - [ ] 8-bit
  - [ ] 4-bit

- [] **Test the inference**
  - [&#10004;] Run inference on a sample image
  - [&#10004;] Compare with PyTorch output
  - [&#10004;] Benchmark inference speed vs. PyTorch for different model sizes

## Done
- [&#10004;] **Image preprocessing**
  - [&#10004;] Load the image from a file name
  - [&#10004;] Create image patches

- [&#10004;] **Convert the PyTorch weights**
  - [&#10004;] Use ggml tensor format to load the params
  - [&#10004;] Validate the weights

- [&#10004;] **Create a ViT object**
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

This project was highly inspired by the following projects:
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)