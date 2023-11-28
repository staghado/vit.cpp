// add simple qunatization strategies

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

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

bool vit_model_quantize(const char *fname_inp, const char *fname_out, const int itype)
{

    ggml_type type = GGML_TYPE_Q4_1;

    switch (itype)
    {
    case 2:
        type = GGML_TYPE_Q4_0;
        break;
    case 3:
        type = GGML_TYPE_Q4_1;
        break;
    case 6:
        type = GGML_TYPE_Q5_0;
        break;
    case 7:
        type = GGML_TYPE_Q5_1;
        break;
    case 8:
        type = GGML_TYPE_Q8_0;
        break;
    default:
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype);
        return false;
    };

    auto ctx_clip = vit_model_load(fname_inp, 2);
    const auto &ctx_src = ctx_clip->ctx_gguf;
    const auto &ctx_data = ctx_clip->ctx;

    auto ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_src);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", itype);

    auto fout = std::ofstream(fname_out, std::ios::binary);

    const int n_tensors = gguf_get_n_tensors(ctx_src);

    for (int i = 0; i < n_tensors; ++i)
    {
        const char *name = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor *cur = ggml_get_tensor(ctx_data, name);
        gguf_add_tensor(ctx_out, cur);
    }

    const size_t meta_size = gguf_get_meta_size(ctx_out);
    for (size_t i = 0; i < meta_size; ++i)
    {
        fout.put(0);
    }

    // regexes of tensor names to be quantized
    const std::vector<std::string> k_names = {
        ".*weight",
    };

    std::vector<uint8_t> read_data(512);
    std::vector<uint8_t> work(512);
    std::vector<float> conv_buf(512);
    std::vector<int64_t> hist_all(1 << 4, 0);
    size_t total_size_org = 0;
    size_t total_size_new = 0;

    for (int i = 0; i < n_tensors; ++i)
    {
        const std::string name = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor *cur = ggml_get_tensor(ctx_data, name.c_str());

        enum ggml_type new_type;
        void *new_data;
        size_t new_size;

        bool quantize = false;
        for (const auto &s : k_names)
        {
            if (std::regex_match(name, std::regex(s)))
            {
                quantize = true;
                break;
            }
        }

        // quantize only 2D tensors
        quantize &= (cur->n_dims == 2);

        if (quantize)
        {
            new_type = type;
            const size_t n_elms = ggml_nelements(cur);
            float *f32_data;

            switch (cur->type)
            {
            case GGML_TYPE_F32:
                f32_data = (float *)cur->data;
                break;
            case GGML_TYPE_F16:
                if (conv_buf.size() < n_elms)
                {
                    conv_buf.resize(n_elms);
                }
                for (int j = 0; j < n_elms; ++j)
                {
                    conv_buf[j] = ggml_fp16_to_fp32(((ggml_fp16_t *)cur->data)[j]);
                }
                f32_data = (float *)conv_buf.data();
                break;
            default:
                printf("Please use an input file in f32 or f16\n");
                return false;
            }

            if (work.size() < n_elms * 4)
            {
                work.resize(n_elms * 4);
            }
            new_data = work.data();

            std::vector<int64_t> hist_cur(1 << 4, 0);

            switch (new_type)
            {
            case GGML_TYPE_Q4_0:
            {
                new_size = ggml_quantize_q4_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
            }
            break;
            case GGML_TYPE_Q4_1:
            {
                new_size = ggml_quantize_q4_1(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
            }
            break;
            case GGML_TYPE_Q5_0:
            {
                new_size = ggml_quantize_q5_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
            }
            break;
            case GGML_TYPE_Q5_1:
            {
                new_size = ggml_quantize_q5_1(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
            }
            break;
            case GGML_TYPE_Q8_0:
            {
                new_size = ggml_quantize_q8_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
            }
            break;
            default:
            {
                fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, new_type);
                return false;
            }
            }

            for (int j = 0; j < hist_cur.size(); ++j)
            {
                hist_all[j] += hist_cur[j];
            }
        }
        else
        {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        }
        const size_t orig_size = ggml_nbytes(cur);
        total_size_org += orig_size;
        total_size_new += new_size;
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data, new_size);
        fout.write((const char *)new_data, new_size);
        size_t pad = GGML_PAD(new_size, gguf_get_alignment(ctx_out)) - new_size;
        for (int j = 0; j < pad; ++j)
        {
            fout.put(0);
        }

        printf("%s: n_dims = %d | quantize=%d | size = %f MB -> %f MB\n", name.c_str(), cur->n_dims, quantize,
               orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
    }

    // go back to beginning of file and write the updated metadata
    fout.seekp(0, std::ios::beg);
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(ctx_out, meta.data());
    fout.write((const char *)meta.data(), meta_size);

    fout.close();

    clip_free(ctx_clip);
    gguf_free(ctx_out);

    {
        printf("%s: original size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
        printf("%s: quantized size  = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);

        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); ++i)
        {
            sum_all += hist_all[i];
        }

        printf("%s: hist: ", __func__);
        for (size_t i = 0; i < hist_all.size(); ++i)
        {
            printf("%5.3f ", hist_all[i] / (float)sum_all);
        }
        printf("\n");
    }

    return true;
}