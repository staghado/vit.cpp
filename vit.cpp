#define _USE_MATH_DEFINES        // for M_PI
#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows

#include "./ggml/ggml.h"
#include "./ggml/ggml-alloc.h"
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

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// default ViT hparams (vit_base_patch8_224.augreg2_in21k_ft_in1k from timm)
struct vit_hparams
{
    int32_t hidden_size = 768;
    int32_t num_hidden_layers = 12;
    int32_t num_attention_heads = 12;
    int32_t num_classes = 1000;
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
        fin.read((char *)&hparams.num_hidden_layers, sizeof(hparams.num_hidden_layers));
        fin.read((char *)&hparams.num_attention_heads, sizeof(hparams.num_attention_heads));
        fin.read((char *)&hparams.ftype, sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: hidden_size            = %d\n", __func__, hparams.hidden_size);
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
        const int32_t num_hidden_layers = hparams.num_hidden_layers;
        const int32_t num_attention_heads = hparams.num_attention_heads;
        const int32_t num_classes = hparams.num_classes;

        const int32_t n_img_embd = hparams.n_img_embd();
        const int32_t n_patch_size = hparams.n_patch_size();

        // image encoder
        {
            ctx_size += hidden_size * (n_img_embd * n_img_embd + 1) * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += hidden_size * ggml_type_sizef(GGML_TYPE_F32);

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
        // dig into this more later!
        ctx_size += (8 + 14 * num_hidden_layers) * ggml_tensor_overhead();

        // classifier
        {
            ctx_size += 2 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += num_classes * hidden_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += num_classes * ggml_type_sizef(GGML_TYPE_F32);
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
    {
        const auto &hparams = model.hparams;

        const int32_t hidden_size = hparams.hidden_size;
        const int32_t num_hidden_layers = hparams.num_hidden_layers;
        const int32_t num_attention_heads = hparams.num_attention_heads;
        const int32_t num_classes = hparams.num_classes;

        const int32_t n_img_embd = hparams.n_img_embd();
        const int32_t n_patch_size = hparams.n_patch_size();

        model.enc_img.layers.resize(num_hidden_layers);

        // image encoder
        {
            auto &enc = model.enc_img;

            enc.pe = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, n_img_embd * n_img_embd + 1, 1);
            enc.cls_token = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, hidden_size);

            enc.proj_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_patch_size, n_patch_size, 3, hidden_size);
            enc.proj_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, hidden_size);

            model.tensors["pos_embed"] = enc.pe;
            model.tensors["cls_token"] = enc.cls_token;

            model.tensors["patch_embed.proj.weight"] = enc.proj_w;
            model.tensors["patch_embed.proj.bias"] = enc.proj_b;

            for (int i = 0; i < num_hidden_layers; ++i)
            {
                auto &layer = enc.layers[i];

                layer.norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
                layer.norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

                layer.qkv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, 3 * hidden_size);
                layer.qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * hidden_size);

                layer.proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, hidden_size);
                layer.proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

                layer.norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
                layer.norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

                layer.mlp_lin1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, 4 * hidden_size);
                layer.mlp_lin1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * hidden_size);

                layer.mlp_lin2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4 * hidden_size, hidden_size);
                layer.mlp_lin2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

                model.tensors["blocks." + std::to_string(i) + ".norm1.weight"] = layer.norm1_w;
                model.tensors["blocks." + std::to_string(i) + ".norm1.bias"] = layer.norm1_b;

                model.tensors["blocks." + std::to_string(i) + ".attn.qkv.weight"] = layer.qkv_w;
                model.tensors["blocks." + std::to_string(i) + ".attn.qkv.bias"] = layer.qkv_b;

                model.tensors["blocks." + std::to_string(i) + ".attn.proj.weight"] = layer.proj_w;
                model.tensors["blocks." + std::to_string(i) + ".attn.proj.bias"] = layer.proj_b;

                model.tensors["blocks." + std::to_string(i) + ".norm2.weight"] = layer.norm2_w;
                model.tensors["blocks." + std::to_string(i) + ".norm2.bias"] = layer.norm2_b;

                model.tensors["blocks." + std::to_string(i) + ".mlp.fc1.weight"] = layer.mlp_lin1_w;
                model.tensors["blocks." + std::to_string(i) + ".mlp.fc1.bias"] = layer.mlp_lin1_b;

                model.tensors["blocks." + std::to_string(i) + ".mlp.fc2.weight"] = layer.mlp_lin2_w;
                model.tensors["blocks." + std::to_string(i) + ".mlp.fc2.bias"] = layer.mlp_lin2_b;
            }
        }

        // classifier
        {
            auto &classifier = model.classifier;

            classifier.norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            classifier.norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            classifier.head_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, num_classes);
            classifier.head_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_classes);

            model.tensors["norm.weight"] = classifier.norm_w;
            model.tensors["norm.bias"] = classifier.norm_b;

            model.tensors["head.weight"] = classifier.head_w;
            model.tensors["head.bias"] = classifier.head_b;
        }

        // load weights
        {
            int n_tensors = 0;
            size_t total_size = 0;

            fprintf(stderr, "%s: ", __func__);

            while (true)
            {
                int32_t n_dims;
                int32_t length;
                int32_t ftype;

                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                fin.read(reinterpret_cast<char *>(&length), sizeof(length));
                fin.read(reinterpret_cast<char *>(&ftype), sizeof(ftype));

                if (fin.eof())
                {
                    break;
                }

                int64_t nelements = 1;
                int64_t ne[4] = {1, 1, 1, 1};
                for (int i = 0; i < n_dims; ++i)
                {
                    int32_t ne_cur;
                    fin.read(reinterpret_cast<char *>(&ne_cur), sizeof(ne_cur));
                    ne[i] = ne_cur;
                    nelements *= ne[i];
                }

                std::string name(length, 0);
                fin.read(&name[0], length);

                if (model.tensors.find(name.data()) == model.tensors.end())
                {
                    fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                    return false;
                }

                auto tensor = model.tensors[name.data()];
                // printf("ne0 = %jd, ne1 = %jd, ne2 = %jd, ne3 = %jd\n", ne[0], ne[1], ne[2], ne[3]);

                if (ggml_nelements(tensor) != nelements)
                {
                    fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %d, expected %d\n",
                            __func__, name.data(), (int)nelements, (int)ggml_nelements(tensor));
                    return false;
                }

                if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2] || tensor->ne[3] != ne[3])
                {
                    fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]\n",
                            __func__, name.data(),
                            (int)ne[0], (int)ne[1], (int)ne[2], (int)ne[3],
                            (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2], (int)tensor->ne[3]);
                    return false;
                }

                size_t bpe = 0;

                switch (ftype)
                {
                case 0:
                    bpe = ggml_type_size(GGML_TYPE_F32);
                    break;
                case 1:
                    bpe = ggml_type_size(GGML_TYPE_F16);
                    break;
                case 2:
                    bpe = ggml_type_size(GGML_TYPE_Q4_0);
                    assert(ne[0] % 64 == 0);
                    break;
                case 3:
                    bpe = ggml_type_size(GGML_TYPE_Q4_1);
                    assert(ne[0] % 64 == 0);
                    break;
                default:
                {
                    fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                    return false;
                }
                };

                if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
                {
                    fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                            __func__, name.data(), ggml_nbytes(tensor), (size_t)nelements * bpe);
                    return false;
                }

                fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

                total_size += ggml_nbytes(tensor);
                if (++n_tensors % 8 == 0)
                {
                    fprintf(stderr, ".");
                    fflush(stdout);
                }
            }

            if (n_tensors != int(model.tensors.size()))
            {
                fprintf(stderr, "%s: model file has %d tensors, but %d tensors were expected\n", __func__, n_tensors, (int)model.tensors.size());
                return false;
            }

            fprintf(stderr, " done\n");

            fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
        }

        fin.close();

        return true;
    }
}

// main function
int main()
{
    const int64_t t_main_start_us = ggml_time_us();

    vit_hparams params;
    std::string filename = "../assets/image.png";
    image_u8 img0;
    image_f32 img1;

    vit_model model;
    vit_state state;
    int64_t t_load_us = 0;

    // load the image
    if (!load_image_from_file(filename.c_str(), img0))
    {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, filename.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, filename.c_str(), img0.nx, img0.ny);

    // preprocess to f32
    if (vit_image_preprocess(img0, img1, params))
    {
        fprintf(stderr, "processed, out dims : (%d x %d)\n", img1.nx, img1.ny);
    }

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!vit_model_load(filename.c_str(), model))
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, filename.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }
    return 0;
}