#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // load image from file using STB open source implementation
#include "./ggml/ggml.h"
#include "./ggml/ggml-alloc.h"

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

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// default ViT hparams (vit_base_patch8_224.augreg2_in21k_ft_in1k from timm)
struct vit_hparams
{
    int32_t hidden_size = 768;
    int32_t intermediate_size = 3072;
    int32_t num_hidden_layers = 12;
    int32_t num_attention_heads = 12;
    int32_t patch_size = 8;
    int32_t img_size = 224;
    int32_t ftype = 1;
    float eps = 1e-6f;

    int32_t n_img_size() const { return 224; }
    int32_t n_patch_size() const { return 8; }
    int32_t n_img_embd() const { return n_img_size() / n_patch_size(); }
};

struct vit_block
{
    struct ggml_tensor *norm1_w;
    struct ggml_tensor *norm1_b;

    struct ggml_tensor *qkv_w;
    struct ggml_tensor *qkv_b;

    struct ggml_tensor *proj_w;
    struct ggml_tensor *proj_b;

    struct ggml_tensor *norm2_w;
    struct ggml_tensor *norm2_b;

    struct ggml_tensor *mlp_lin1_w;
    struct ggml_tensor *mlp_lin1_b;

    struct ggml_tensor *mlp_lin2_w;
    struct ggml_tensor *mlp_lin2_b;
};

struct vit_model
{
    vit_hparams hparams;

    struct ggml_context *ctx;

    struct ggml_tensor *pe;
    struct ggml_tensor *cls_token;

    std::vector<vit_block> layers;
};

// image loading
// RGB uint8 image
struct image_u8
{
    int nx;
    int ny;

    std::vector<uint8_t> data;
};

// RGB float32 image
// Memory layout: RGBRGBRGB...
struct image_f32
{
    int nx;
    int ny;

    std::vector<float> data;
};

// load image from a file(uses stbi_load)
bool load_image_from_file(const std::string &fname, image_u8 &img)
{
    int nx, ny, nc;
    auto data = stbi_load(fname.c_str(), &nx, &ny, &nc, 3);
    if (!data)
    {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }

    img.nx = nx;
    img.ny = ny;
    img.data.resize(nx * ny * 3);
    memcpy(img.data.data(), data, nx * ny * 3);

    stbi_image_free(data);

    return true;
}

// load the model's weights from a file
bool vit_model_load(const std::string &fname, vit_model &model)
{
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    auto &ctx = model.ctx;

    size_t ctx_size = 0;

    // {
    //     const auto &hparams = model.hparams;

    //     const int n_input = hparams.n_input;
    //     const int n_hidden = hparams.n_hidden;
    //     const int n_classes = hparams.n_classes;

    //     ctx_size += n_input * n_hidden * ggml_type_sizef(GGML_TYPE_F32); // fc1 weight
    //     ctx_size += n_hidden * ggml_type_sizef(GGML_TYPE_F32);           // fc1 bias

    //     ctx_size += n_hidden * n_classes * ggml_type_sizef(GGML_TYPE_F32); // fc2 weight
    //     ctx_size += n_classes * ggml_type_sizef(GGML_TYPE_F32);            // fc2 bias

    //     printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));
    // }

    // // create the ggml context
    // {
    //     struct ggml_init_params params = {
    //         /*.mem_size   =*/ctx_size + 1024 * 1024,
    //         /*.mem_buffer =*/NULL,
    //         /*.no_alloc   =*/false,
    //     };

    //     model.ctx = ggml_init(params);
    //     if (!model.ctx)
    //     {
    //         fprintf(stderr, "%s: ggml_init() failed\n", __func__);
    //         return false;
    //     }
    // }

    // // Read FC1 layer 1
    // {
    //     // Read dimensions
    //     int32_t n_dims;
    //     fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));

    //     {
    //         int32_t ne_weight[2] = {1, 1};
    //         for (int i = 0; i < n_dims; ++i)
    //         {
    //             fin.read(reinterpret_cast<char *>(&ne_weight[i]), sizeof(ne_weight[i]));
    //         }

    //         // FC1 dimensions taken from file, eg. 768x500
    //         model.hparams.n_input = ne_weight[0];
    //         model.hparams.n_hidden = ne_weight[1];

    //         model.fc1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.n_input, model.hparams.n_hidden);
    //         fin.read(reinterpret_cast<char *>(model.fc1_weight->data), ggml_nbytes(model.fc1_weight));
    //         ggml_set_name(model.fc1_weight, "fc1_weight");
    //     }

    //     {
    //         int32_t ne_bias[2] = {1, 1};
    //         for (int i = 0; i < n_dims; ++i)
    //         {
    //             fin.read(reinterpret_cast<char *>(&ne_bias[i]), sizeof(ne_bias[i]));
    //         }

    //         model.fc1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_hidden);
    //         fin.read(reinterpret_cast<char *>(model.fc1_bias->data), ggml_nbytes(model.fc1_bias));
    //         ggml_set_name(model.fc1_bias, "fc1_bias");

    //         // just for testing purposes, set some parameters to non-zero
    //         model.fc1_bias->op_params[0] = 0xdeadbeef;
    //     }
    // }

    // fin.close();

    return true;
}

// main function
// int main(int argc, char **argv)
// {
//     srand(time(NULL));
//     // ggml_time_init();

//     if (argc != 3)
//     {
//         fprintf(stderr, "Usage: %s models/mnist/ggml-model-f32.bin image.jpg\n", argv[0]);
//         exit(0);
//     }

//     uint8_t buf[784];
//     vit_model model;
//     std::vector<float> digit;

//     // load the model
//     // {
//     //     const int64_t t_start_us = ggml_time_us();

//     //     if (!vit_model_load(argv[1], model))
//     //     {
//     //         fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, "models/ggml-model-f32.bin");
//     //         return 1;
//     //     }

//     //     const int64_t t_load_us = ggml_time_us() - t_start_us;

//     //     fprintf(stdout, "%s: loaded model in %8.2f ms\n", __func__, t_load_us / 1000.0f);
//     // }

//     // ggml_free(model.ctx);

//     return 0;
// }

int main()
{
    // load the image
    std::string filename = "../assets/image.png";
    image_u8 img0;
    if (!load_image_from_file(filename.c_str(), img0))
    {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, filename.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, filename.c_str(), img0.nx, img0.ny);

    return 0;
}