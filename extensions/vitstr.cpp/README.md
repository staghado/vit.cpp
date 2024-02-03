# ViTSTR: Vision Transformer for Fast and Efficient Scene Text Recognition
[Code](https://github.com/roatienza/deep-text-recognition-benchmark)


## Build

    mkdir -p build && cd build
    cmake ..
    make -j4


## Usage


<p align="center">
<img src="images/demo_1.png" alt="example input" width="50%" height="auto">
</p>

<pre>
./bin/vitstr -t 4 -m ../ggml-model-f16.gguf -i ../images/demo_1.png 
main: seed = 1706997535
main: n_threads = 4 / 8
vit_model_load: loading model from '../ggml-model-f16.gguf' - please wait
vit_model_load: hidden_size            = 768
vit_model_load: num_hidden_layers      = 12
vit_model_load: num_attention_heads    = 12
vit_model_load: patch_size             = 16
vit_model_load: img_size               = 224
vit_model_load: num_classes            = 96
vit_model_load: ftype                  = 1
vit_model_load: qntvr                  = 0
operator(): ggml ctx size = 164.48 MB
vit_model_load: ................... done
vit_model_load: model size =   163.56 MB / num tensors = 152
main: loaded image '../images/demo_1.png' (184 x 72)
processed, out dims : (224 x 224)
------------------ 
Available
score : 1.00 
------------------ 


main:    model load time =   144.64 ms
main:    processing time =  1176.77 ms
main:    total time      =  1321.41 ms 
