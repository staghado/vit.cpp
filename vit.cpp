#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // load image from file using STB open source implementation

#include <cassert>   // provides assertion
#include <cmath>     // sin, cos, M_PI
#include <cstddef>   // defines size_t
#include <cstdio>    // IO functions like printf, scanf, etc
#include <cstring>   // string maanipulation like strcpy(), strlen(), and strcmp()
#include <fstream>   // read from and write to files
#include <map>       // key-value container
#include <string>    // more flexible strings than C-style 
#include <vector>    // standard 'vector' dynamic container
#include <thread>    // manage threads
#include <cinttypes> // format macros for integer types across-platforms like PRId64, PRIu32, etc


// default ViT hparams (vit_base_patch8_224.augreg2_in21k_ft_in1k from timm)
struct vit_hparams {
    int32_t hidden_size = 768;
    int32_t intermediate_size = 3072;
    int32_t num_hidden_layers = 12;
    int32_t num_attention_heads = 12;
    int32_t patch_size = 8;
    int32_t img_size = 224;
    int32_t ftype = 1;
    float eps = 1e-6f;

    int32_t n_img_size()     const { return 224; }
    int32_t n_patch_size()   const { return 8; }
    int32_t n_img_embd()     const { return n_img_size() / n_patch_size(); }

};
