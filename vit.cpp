#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#define STB_IMAGE_IMPLEMENTATION
#include "ggml/examples/stb_image.h" // stb image load

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <cinttypes>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// default ViT-B hparams
// vit_base_patch8_224.augreg2_in21k_ft_in1k from timm
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

    // id2label map
    std::map<int, std::string> id2label;

    int32_t n_enc_head_dim() const { return hidden_size / num_attention_heads; }
    int32_t n_img_size() const { return img_size; }
    int32_t n_patch_size() const { return patch_size; }
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
    // layer norm
    struct ggml_tensor *norm_w;
    struct ggml_tensor *norm_b;

    // head
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
    struct ggml_tensor *prediction;
    struct ggml_context *ctx;

    // buffer for `ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // buffers to evaluate the model
    std::vector<uint8_t> buf_alloc_img_enc;
    std::vector<uint8_t> buf_compute_img_enc;

    struct ggml_allocr *allocr = {};
};

struct vit_model
{
    // hparams
    vit_hparams hparams;

    // model
    vit_image_encoder enc_img;
    classifier_head classifier;

    // context
    struct ggml_context *ctx;

    // tensor map
    std::map<std::string, struct ggml_tensor *> tensors;
};

//
//  Image loading and preprocessing
//

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

//
// Helpers
//

void print_t_f32(const char *title, struct ggml_tensor *t, int n = 10)
{
    printf("%s\n", title);
    float *data = (float *)t->data;
    printf("dims: % " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " f32\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    printf("First & Last %d elements:\n", n);
    for (int i = 0; i < std::min((int)(t->ne[0] * t->ne[1]), n); i++)
    {
        printf("%.5f ", data[i]);
        if (i != 0 && i % t->ne[0] == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
    for (int i = 0; i < std::min((int)(t->ne[0] * t->ne[1]), n); i++)
    {
        printf("%.5f ", data[ggml_nelements(t) - n + i]);
        if ((ggml_nelements(t) - n + i) % t->ne[0] == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
    double sum = 0.0;
    for (int i = 0; i < ggml_nelements(t); i++)
    {
        sum += data[i];
    }
    printf("sum:  %f\n\n", sum);
}

static void ggml_disconnect_node_from_graph(ggml_tensor *t)
{
    t->op = GGML_OP_NONE;
    for (int i = 0; i < GGML_MAX_SRC; i++)
    {
        t->src[i] = NULL;
    }
}

static void ggml_graph_compute_helper(std::vector<uint8_t> &buf, ggml_cgraph *graph, int n_threads)
{
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0)
    {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

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
// TO DO : use bicubic interpolation instead of bilinear
bool vit_image_preprocess(const image_u8 &img, image_f32 &res, const vit_hparams &params)
{
    const int nx = img.nx;
    const int ny = img.ny;

    const int nx2 = params.n_img_size();
    const int ny2 = params.n_img_size();
    res.nx = nx2;
    res.ny = ny2;
    res.data.resize(3 * nx2 * ny2);

    const float scale = std::max(nx, ny) / (float)params.n_img_size();

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

// load the model's weights from a file following the ggml format(gguf)
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
        fin.read((char *)&hparams.num_classes, sizeof(hparams.num_classes));
        fin.read((char *)&hparams.patch_size, sizeof(hparams.patch_size));
        fin.read((char *)&hparams.img_size, sizeof(hparams.img_size));
        fin.read((char *)&hparams.ftype, sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: hidden_size            = %d\n", __func__, hparams.hidden_size);
        printf("%s: num_hidden_layers      = %d\n", __func__, hparams.num_hidden_layers);
        printf("%s: num_attention_heads    = %d\n", __func__, hparams.num_attention_heads);
        printf("%s: patch_size             = %d\n", __func__, hparams.patch_size);
        printf("%s: img_size               = %d\n", __func__, hparams.img_size);
        printf("%s: num_classes            = %d\n", __func__, hparams.num_classes);
        printf("%s: ftype                  = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr                  = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;

        // read id2label dictionary into an ordered map (sort of an OrderedDict)
        int num_labels;
        fin.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));

        for (int i = 0; i < num_labels; ++i)
        {
            int key;
            int value_length;
            fin.read(reinterpret_cast<char *>(&key), sizeof(key));
            fin.read(reinterpret_cast<char *>(&value_length), sizeof(value_length));

            std::string value(value_length, '\0');
            fin.read(&value[0], value_length);

            model.hparams.id2label[key] = value;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    // ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
    // if (wtype == GGML_TYPE_COUNT)
    // {
    //     fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
    //             __func__, fname.c_str(), model.hparams.ftype);
    //     return false;
    // }

    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.ftype)
    {
    case 0:
        wtype = GGML_TYPE_F32;
        break;
    case 1:
        wtype = GGML_TYPE_F16;
        break;
    case 2:
        wtype = GGML_TYPE_Q4_0;
        break;
    case 3:
        wtype = GGML_TYPE_Q4_1;
        break;
    case 6:
        wtype = GGML_TYPE_Q5_0;
        break;
    case 7:
        wtype = GGML_TYPE_Q5_1;
        break;
    case 8:
        wtype = GGML_TYPE_Q8_0;
        break;
    default:
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }
    }

    auto &ctx = model.ctx;

    // lambda function to calculate ggml context
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
            ctx_size += hidden_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += hidden_size * (n_img_embd * n_img_embd + 1) * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += hidden_size * 3 * n_patch_size * n_patch_size * ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += hidden_size * ggml_type_sizef(GGML_TYPE_F32);
        }

        // image encoder layers
        {
            ctx_size += num_hidden_layers * hidden_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += num_hidden_layers * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += num_hidden_layers * 3 * hidden_size * hidden_size * ggml_type_sizef(wtype);
            ctx_size += num_hidden_layers * 3 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += num_hidden_layers * hidden_size * hidden_size * ggml_type_sizef(wtype);
            ctx_size += num_hidden_layers * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += num_hidden_layers * hidden_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += num_hidden_layers * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += num_hidden_layers * 4 * hidden_size * hidden_size * ggml_type_sizef(wtype);
            ctx_size += num_hidden_layers * 4 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += num_hidden_layers * 4 * hidden_size * hidden_size * ggml_type_sizef(wtype);
            ctx_size += num_hidden_layers * 4 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);
        }

        // dig into this more later!
        ctx_size += (8 + 14 * num_hidden_layers) * ggml_tensor_overhead();

        // classifier
        {
            ctx_size += 2 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += num_classes * hidden_size * ggml_type_sizef(wtype);
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
            enc.cls_token = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, 1, 1);

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

                layer.qkv_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, 3 * hidden_size);
                layer.qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * hidden_size);

                layer.proj_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
                layer.proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

                layer.norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
                layer.norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

                layer.mlp_lin1_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, 4 * hidden_size);
                layer.mlp_lin1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * hidden_size);

                layer.mlp_lin2_w = ggml_new_tensor_2d(ctx, wtype, 4 * hidden_size, hidden_size);
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

            classifier.head_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, num_classes);
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
                case 6:
                    bpe = ggml_type_size(GGML_TYPE_Q5_0);
                    assert(ne[0] % 64 == 0);
                    break;
                case 7:
                    bpe = ggml_type_size(GGML_TYPE_Q5_1);
                    assert(ne[0] % 64 == 0);
                    break;
                case 8:
                    bpe = ggml_type_size(GGML_TYPE_Q8_0);
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

//
// ViT Encoder
//

struct ggml_cgraph *vit_encode_image(
    const vit_model &model,
    vit_state &state,
    const image_f32 &img)
{

    const auto &hparams = model.hparams;
    const auto &enc = model.enc_img;
    const auto &classifier = model.classifier;

    const int32_t hidden_size = hparams.hidden_size;
    const int32_t num_hidden_layers = hparams.num_hidden_layers;
    const int32_t num_attention_heads = hparams.num_attention_heads;
    const int32_t num_classes = hparams.num_classes;
    const int32_t n_img_size = hparams.img_size;
    const int32_t n_enc_head_dim = hparams.n_enc_head_dim();

    const int32_t n_img_embd = hparams.n_img_embd();
    const int32_t n_patch_size = hparams.n_patch_size();

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/state.buf_compute_img_enc.size(),
        /*.mem_buffer =*/state.buf_compute_img_enc.data(),
        /*.no_alloc   =*/true, // skip allocating as we use ggml_alloc to allocate exact memory requirements
    };

    struct ggml_context *ctx0 = ggml_init(ggml_params);
    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    struct ggml_tensor *inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_img_size, n_img_size, 3, 1);
    ggml_allocr_alloc(state.allocr, inp);
    if (!ggml_allocr_is_measure(state.allocr))
    {
        float *data = (float *)ggml_get_data(inp);

        const int nx = img.nx;
        const int ny = img.ny;
        const int n = nx * ny;

        GGML_ASSERT(nx == n_img_size && ny == n_img_size);

        for (int k = 0; k < 3; k++)
        {
            for (int y = 0; y < ny; y++)
            {
                for (int x = 0; x < nx; x++)
                {
                    data[k * n + y * nx + x] = img.data[3 * (y * nx + x) + k];
                }
            }
        }
    }

    // patch embedding
    struct ggml_tensor *cur = ggml_conv_2d_sk_p0(ctx0, enc.proj_w, inp);
    cur = ggml_add_inplace(ctx0,
                           cur,
                           ggml_repeat(ctx0, enc.proj_b, cur));

    // keep in F32
    cur = ggml_cont(ctx0,
                    ggml_permute(ctx0, cur, 1, 2, 0, 3));

    // convert to F16
    // cur = ggml_cpy(ctx0,
    //                ggml_permute(ctx0, cur, 1, 2, 0, 3),
    //                ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, hidden_size, n_img_embd, n_img_embd));

    // add positional embedding
    // cur dim     : 768  28  28  1
    // enc.pe dim  : 768  785  1  1

    // reshape patch embeddings from (768  28  28  1) to (768  784  1  1)
    cur = ggml_reshape_4d(ctx0, cur, hidden_size, n_img_embd * n_img_embd, 1, 1);

    // concat class embeddings(cls_token) : (768  1  1  1) with positional embeddings (pos_embed = cur) : (768  784  1  1)
    cur = ggml_permute(ctx0, ggml_concat(ctx0, enc.cls_token, ggml_permute(ctx0, cur, 0, 2, 1, 3)),
                       0, 2, 1, 3); // 768  785  1  1

    cur = ggml_add_inplace(ctx0, cur, enc.pe);

    struct ggml_tensor *inpL = cur;

    // loop over layers
    for (int il = 0; il < num_hidden_layers; ++il)
    {
        const auto &layer = enc.layers[il];

        // norm 1
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);

            // cur = w * cur + b
            cur = ggml_mul(ctx0, cur, layer.norm1_w);
            cur = ggml_add_inplace(ctx0, cur, layer.norm1_b);
        }

        const int64_t W = cur->ne[1];
        const int64_t H = cur->ne[2];

        // self-attention
        {
            cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.qkv_b);

            // split qkv into separate tensors
            const int B = cur->ne[3];

            cur = ggml_reshape_4d(ctx0, cur, hidden_size, 3, W * H, B);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 3, 1, 2));

            struct ggml_tensor *Q;
            struct ggml_tensor *K;
            struct ggml_tensor *V;

            Q = ggml_view_3d(ctx0, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2], 0 * cur->nb[3]);
            Q = ggml_reshape_4d(ctx0, Q, n_enc_head_dim, num_attention_heads, W * H, B);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, n_enc_head_dim, W * H, B * num_attention_heads);

            K = ggml_view_3d(ctx0, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
            K = ggml_reshape_4d(ctx0, K, n_enc_head_dim, num_attention_heads, W * H, B);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, n_enc_head_dim, W * H, B * num_attention_heads);

            V = ggml_view_3d(ctx0, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2], 2 * cur->nb[3]);
            V = ggml_reshape_4d(ctx0, V, n_enc_head_dim, num_attention_heads, W * H, B);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // transposed
            V = ggml_reshape_3d(ctx0, V, W * H, n_enc_head_dim, B * num_attention_heads);

            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

            // attention weights
            struct ggml_tensor *KQ_scaled =
                ggml_scale_inplace(ctx0,
                                   KQ,
                                   ggml_new_f32(ctx0, 1.0f / sqrtf(n_enc_head_dim)));

            struct ggml_tensor *KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_scaled);

            struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            cur =
                ggml_reshape_4d(ctx0,
                                ggml_cont(ctx0,
                                          ggml_permute(ctx0,
                                                       ggml_reshape_4d(ctx0, KQV, n_enc_head_dim, W * H, num_attention_heads, B),
                                                       0, 2, 1, 3)),
                                hidden_size, W, H, B);

            cur = ggml_mul_mat(ctx0, layer.proj_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.proj_b);
        }

        // add skip connection
        cur = ggml_add_inplace(ctx0, cur, inpL);

        struct ggml_tensor *inpFF = cur;

        // feed-forward network
        {
            // norm 2
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = w * cur + b
                cur = ggml_mul(ctx0, cur, layer.norm2_w);
                cur = ggml_add_inplace(ctx0, cur, layer.norm2_b);
            }

            // fully connected layer
            cur = ggml_mul_mat(ctx0, layer.mlp_lin1_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.mlp_lin1_b);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0, layer.mlp_lin2_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.mlp_lin2_b);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;

    //
    // pooling
    //

    // get the output of cls token at index 0
    struct ggml_tensor *cls = ggml_new_i32(ctx0, 0);
    cur = ggml_get_rows(ctx0, cur, cls);

    // layer normalization
    {
        cur = ggml_norm(ctx0, cur, hparams.eps);

        // cur = w * cur + b
        cur = ggml_mul(ctx0, cur, classifier.norm_w);
        cur = ggml_add_inplace(ctx0, cur, classifier.norm_b);
    }

    //
    // classification head
    //

    // projection
    cur = ggml_mul_mat(ctx0, classifier.head_w, cur);
    cur = ggml_add_inplace(ctx0, cur, classifier.head_b);

    // softmax
    ggml_tensor *probs = ggml_soft_max(ctx0, cur);

    probs = ggml_cpy(ctx0, probs, state.prediction);

    ggml_build_forward_expand(gf, probs);
    ggml_disconnect_node_from_graph(state.prediction);

    ggml_free(ctx0);

    return gf;
}

struct vit_params
{
    int32_t seed = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t topk = 5;

    std::string model = "../ggml-model-f16.gguf";  // model path
    std::string fname_inp = "../assets/tench.jpg"; // image path
    float eps = 1e-6f;                             // epsilon used in LN
};

void print_usage(int argc, char **argv, const vit_params &params)
{
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help              show this help message and exit\n");
    fprintf(stderr, "  -m FNAME, --model       model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --inp         input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -t N, --threads         number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -k N, --topk            top k classes to print (default: %d)\n", params.topk);
    fprintf(stderr, "  -s SEED, --seed         RNG seed (default: -1)\n");
    fprintf(stderr, "  -e FLOAT, --epsilon     epsilon constant in Layer Norm layers (default: %f)\n", params.eps);
    fprintf(stderr, "\n");
}

bool vit_params_parse(int argc, char **argv, vit_params &params)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed")
        {
            params.seed = std::stoi(argv[++i]);
        }
        else if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-i" || arg == "--inp")
        {
            params.fname_inp = argv[++i];
        }
        else if (arg == "-k" || arg == "--topk")
        {
            params.topk = std::stoi(argv[++i]);
        }
        else if (arg == "-e" || arg == "--epsilon")
        {
            params.eps = std::stof(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_usage(argc, argv, params);
            exit(0);
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

// main function
int main(int argc, char **argv)
{
    const int64_t t_main_start_us = ggml_time_us();

    vit_params params;

    image_u8 img0;
    image_f32 img1;

    vit_model model;
    vit_state state;

    int64_t t_load_us = 0;

    if (vit_params_parse(argc, argv, params) == false)
    {
        return 1;
    }

    if (params.seed < 0)
    {
        params.seed = time(NULL);
    }
    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);
    fprintf(stderr, "%s: n_threads = %d / %d\n", __func__, params.n_threads, (int32_t)std::thread::hardware_concurrency());

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!vit_model_load(params.model.c_str(), model))
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    // load the image
    if (!load_image_from_file(params.fname_inp.c_str(), img0))
    {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img0.nx, img0.ny);

    // preprocess the image to f32
    if (vit_image_preprocess(img0, img1, model.hparams))
    {
        fprintf(stderr, "processed, out dims : (%d x %d)\n", img1.nx, img1.ny);
    }

    // prepare for graph computation, memory allocation and results processing
    {
        static size_t buf_size = 3u * 1024 * 1024;

        struct ggml_init_params ggml_params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/false,
        };

        state.ctx = ggml_init(ggml_params);
        state.prediction = ggml_new_tensor_4d(state.ctx, GGML_TYPE_F32, model.hparams.num_classes, 1, 1, 1);

        // printf("%s: Initialized context = %ld bytes\n", __func__, buf_size);
    }

    static const size_t tensor_alignment = 32;
    {
        // first build the graph and record memory requirements
        state.buf_compute_img_enc.resize(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead());
        state.allocr = ggml_allocr_new_measure(tensor_alignment);
        struct ggml_cgraph *gf_measure = vit_encode_image(model, state, img1);
        if (!gf_measure)
        {
            fprintf(stderr, "%s: failed to encode image\n", __func__);
            return 1;
        }

        size_t alloc_size = ggml_allocr_alloc_graph(state.allocr, gf_measure) + tensor_alignment;
        ggml_allocr_free(state.allocr);

        // recreate allocator with exact memory requirements
        state.buf_alloc_img_enc.resize(alloc_size);
        state.allocr = ggml_allocr_new(state.buf_alloc_img_enc.data(), state.buf_alloc_img_enc.size(), tensor_alignment);

        // compute the graph with the measured exact memory requirements from above
        ggml_allocr_reset(state.allocr);

        struct ggml_cgraph *gf = vit_encode_image(model, state, img1);
        if (!gf)
        {
            fprintf(stderr, "%s: failed to encode image\n", __func__);
            return 1;
        }

        ggml_allocr_alloc_graph(state.allocr, gf);
        ggml_graph_compute_helper(state.work_buffer, gf, params.n_threads);

        // print_t_f32("after probs", state.prediction);

        const float *probs_data = ggml_get_data_f32(state.prediction);

        // top-5
        std::vector<std::pair<float, int>> predictions;

        // store probability and index
        for (int i = 0; i < model.hparams.num_classes; ++i)
        {
            predictions.push_back(std::make_pair(probs_data[i], i));
        }

        // sort in descending order
        std::sort(predictions.begin(), predictions.end(),
                  [](const std::pair<float, int> &a, const std::pair<float, int> &b)
                  {
                      return a.first > b.first;
                  });

        fprintf(stderr, "\n");

        // top k predictions
        for (int i = 0; i < params.topk && i < predictions.size(); ++i)
        {
            printf(" > %s : %.2f\n",
                   model.hparams.id2label[predictions[i].second].c_str(),
                   predictions[i].first);
        }

        // free memory
        ggml_allocr_free(state.allocr);
        state.allocr = NULL;
        state.work_buffer.clear();
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();
        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s:    model load time = %8.2f ms\n", __func__, t_load_us / 1000.0f);
        fprintf(stderr, "%s:    processing time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us - t_load_us) / 1000.0f);
        fprintf(stderr, "%s:    total time      = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
