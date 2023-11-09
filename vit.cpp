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
    printf("%s: loading model from '%s' - please wait\n", __func__, fname.c_str());

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

    // load hparams
    {
        // override defaults
        auto &hparams = model.hparams;

        fin.read((char *)&hparams.hidden_size, sizeof(hparams.hidden_size));
        fin.read((char *)&hparams.intermediate_size, sizeof(hparams.intermediate_size));
        fin.read((char *)&hparams.num_hidden_layers, sizeof(hparams.num_hidden_layers));
        fin.read((char *)&hparams.num_attention_heads, sizeof(hparams.num_attention_heads));
        fin.read((char *)&hparams.ftype, sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: hidden_size            = %d\n", __func__, hparams.hidden_size);
        printf("%s: intermediate_size      = %d\n", __func__, hparams.intermediate_size);
        printf("%s: num_hidden_layers      = %d\n", __func__, hparams.num_hidden_layers);
        printf("%s: num_attention_heads    = %d\n", __func__, hparams.num_attention_heads);
        printf("%s: ftype                  = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr                  = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT)
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    auto &ctx = model.ctx;
    // lambda fucntion to calculate ggml context
    const size_t ctx_size = [&]()
    {
        size_t ctx_size = 0;

        const auto &hparams = model.hparams;

        const int32_t hidden_size = hparams.hidden_size;
        const int32_t intermediate_size = hparams.intermediate_size;
        const int32_t num_hidden_layers = hparams.num_hidden_layers;
        const int32_t num_attention_heads = hparams.num_attention_heads;

        const int32_t n_img_embd = hparams.n_img_embd();
        const int32_t n_patch_size = hparams.n_patch_size();

        // image encoder
        {
            ctx_size += hidden_size * n_img_embd * n_img_embd * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += hidden_size * 3 * n_patch_size * n_patch_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += hidden_size * ggml_type_sizef(GGML_TYPE_F32);
        }

        // image encoder layers
        {
            ctx_size += num_hidden_layers * hidden_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += num_hidden_layers * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += num_hidden_layers * 3 * hidden_size * hidden_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += num_hidden_layers * 3 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += num_hidden_layers * hidden_size * hidden_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += num_hidden_layers * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += num_hidden_layers * hidden_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += num_hidden_layers * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += num_hidden_layers * 4 * hidden_size * hidden_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += num_hidden_layers * 4 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += num_hidden_layers * 4 * hidden_size * hidden_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += num_hidden_layers * 4 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);
        }

        ctx_size += (8 + 14 * num_hidden_layers) * ggml_tensor_overhead();

        // transformer
        {
            const int tfm_layers_count = 2;
            const int qkv_count = 3;
            const int norm_count = 4;
            const int n_hypernet_mpls_count = 4;

            // self_attn
            ctx_size += tfm_layers_count * qkv_count * hidden_size * hidden_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += tfm_layers_count * qkv_count * hidden_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += tfm_layers_count * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            // all norms
            ctx_size += tfm_layers_count * norm_count * hidden_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += tfm_layers_count * norm_count * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            // cross_attn_token_to_img
            ctx_size += tfm_layers_count * qkv_count * hidden_size * (hidden_size / 2) * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += tfm_layers_count * qkv_count * (hidden_size / 2) * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += tfm_layers_count * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            // mlp
            ctx_size += tfm_layers_count * 8 * intermediate_size * intermediate_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += tfm_layers_count * 8 * intermediate_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += tfm_layers_count * intermediate_size * 8 * intermediate_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += tfm_layers_count * intermediate_size * ggml_type_sizef(GGML_TYPE_F32);

            // transformer_norm_final
            ctx_size += norm_count * hidden_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += norm_count * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            // output_upscaling
            ctx_size += intermediate_size * n_img_embd * 2 * 2 * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += 3 * n_img_embd * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += intermediate_size * n_img_embd * (n_img_embd / 2) * 2 * 2 * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += (n_img_embd / 2) * ggml_type_sizef(GGML_TYPE_F32);

            // output_hypernetworks_mlps
            ctx_size += n_hypernet_mpls_count * 2 * intermediate_size * intermediate_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += n_hypernet_mpls_count * 2 * intermediate_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += n_hypernet_mpls_count * intermediate_size * (n_img_embd / 2) * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += n_hypernet_mpls_count * (n_img_embd / 2) * ggml_type_sizef(GGML_TYPE_F32);

            // classification head
            ctx_size += 2 * intermediate_size * intermediate_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += 2 * intermediate_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += 1000 * hidden_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += 1000 * ggml_type_sizef(GGML_TYPE_F32);
        }

        fprintf(stderr, "%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));

        return ctx_size;
    }();

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ctx_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/false,
        };

        ctx = ggml_init(params);
        if (!ctx)
        {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights

    // load weights

    return true;
}

// main function
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