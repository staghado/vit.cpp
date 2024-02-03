#pragma once

// extensions/vitstr.cpp/convert-pth-to-ggml.py
#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
// #include "ggml/examples/stb_image.h"
#include "/home/said/projects/vit.cpp/ggml/examples/stb_image.h" // stb image load

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
    std::map<int, std::string> id2label;

    int32_t n_enc_head_dim() const;
    int32_t n_img_size() const;
    int32_t n_patch_size() const;
    int32_t n_img_embd() const;
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
    struct ggml_tensor *norm_w;
    struct ggml_tensor *norm_b;
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
    std::vector<uint8_t> work_buffer;
    std::vector<uint8_t> buf_alloc_img_enc;
    std::vector<uint8_t> buf_compute_img_enc;
    struct ggml_allocr *allocr;
};

struct vit_model
{
    vit_hparams hparams;
    vit_image_encoder enc_img;
    classifier_head classifier;
    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct image_u8
{
    int nx;
    int ny;
    std::vector<uint8_t> data;
};

struct image_f32
{
    int nx;
    int ny;
    std::vector<float> data;
};

struct vit_params
{
    int32_t seed = -1;
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t topk = 5;
    std::string model = "../ggml-model-f16.gguf";  // model path
    std::string fname_inp = "../assets/tench.jpg"; // image path
    float eps = 1e-6f;                             // epsilon used in LN
};

void print_t_f32(const char *title, struct ggml_tensor *t, int n);
static void ggml_disconnect_node_from_graph(ggml_tensor *t);
void ggml_graph_compute_helper(std::vector<uint8_t> &buf, ggml_cgraph *graph, int n_threads);
bool load_image_from_file(const std::string &fname, image_u8 &img);
bool vit_image_preprocess(const image_u8 &img, image_f32 &res, const vit_hparams &params);
bool vit_model_load(const std::string &fname, vit_model &model);
struct ggml_cgraph *vit_encode_image(const vit_model &model, vit_state &state, const image_f32 &img);
int vit_predict(const vit_model &model, vit_state &state, const image_f32 img1, const vit_params &params, std::vector<std::pair<float, int>> &predictions);
void print_usage(int argc, char **argv, const vit_params &params);
bool vit_params_parse(int argc, char **argv, vit_params &params);