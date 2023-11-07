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

struct classifier_head
{
    // Layer norm
    struct ggml_tensor *norm_w;
    struct ggml_tensor *norm_b;

    // Head
    struct ggml_tensor *head_w;
    struct ggml_tensor *head_b;
};

struct vit_image_encoder
{
    struct ggml_tensor *pe;
    struct ggml_tensor *cls_token;

    struct ggml_tensor *proj_w;
    struct ggml_tensor *proj_b;

    std::vector<vit_block> layers;
};

struct vit_state
{
    struct ggml_tensor *embd_img;

    struct ggml_context *ctx;

    // buffer for `ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;
    // buffers to evaluate the model
    std::vector<uint8_t> buf_alloc_img_enc;
    std::vector<uint8_t> buf_compute_img_enc;

    std::vector<uint8_t> buf_alloc_fast;
    std::vector<uint8_t> buf_compute_fast;

    struct ggml_allocr *allocr = {};
};

struct vit_model
{
    vit_hparams hparams;

    vit_image_encoder enc_img;
    classifier_head classifier;

    // context
    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
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

// preprocess input image : resize + normalize
bool vit_image_preprocess(const image_u8 &img, image_f32 &res, const vit_hparams &params)
{
    const int nx = img.nx;
    const int ny = img.ny;

    const int nx2 = params.n_img_size();
    const int ny2 = params.n_img_size();
    res.nx = nx2;
    res.ny = ny2;
    res.data.resize(3 * nx2 * ny2);

    const float scale = std::max(nx, ny) / 224.0f;

    fprintf(stderr, "%s: scale = %f\n", __func__, scale);

    const int nx3 = int(nx / scale + 0.5f);
    const int ny3 = int(ny / scale + 0.5f);

    const float m3[3] = {123.675f, 116.280f, 103.530f};
    const float s3[3] = {58.395f, 57.120f, 57.375f};

    for (int y = 0; y < ny3; y++)
    {
        for (int x = 0; x < nx3; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                // linear interpolation
                const float sx = (x + 0.5f) * scale - 0.5f;
                const float sy = (y + 0.5f) * scale - 0.5f;

                const int x0 = std::max(0, (int)std::floor(sx));
                const int y0 = std::max(0, (int)std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = img.data[j00];
                const float v01 = img.data[j01];
                const float v10 = img.data[j10];
                const float v11 = img.data[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res.data[i] = (float(v2) - m3[c]) / s3[c];
            }
        }
    }

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

    return true;
}

// main function
// int main(int argc, char **argv)
// {
//     srand(time(NULL));
//     // ggml_time_init();

//     if (argc != 3)
//     {
//         fprintf(stderr, "Usage: %s models/ggml-model-f32.bin image.jpg\n", argv[0]);
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
    vit_hparams params;
    std::string filename = "../assets/image.png";
    image_u8 img0;
    image_f32 img1;

    if (!load_image_from_file(filename.c_str(), img0))
    {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, filename.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, filename.c_str(), img0.nx, img0.ny);

    if (vit_image_preprocess(img0, img1, params))
    {
        fprintf(stderr, "processed, out dims : (%d x %d)\n", img1.nx, img1.ny);
    }
    return 0;
}