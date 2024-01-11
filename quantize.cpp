// add simple qunatization strategies
// adapted from : ggml/gpt-2

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex>

// default ViT-B hparams
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
    std::string interpolation = "bicubic";
    // id2label map
    std::map<int, std::string> id2label;
};

// quantize a model
bool vit_model_quantize(const std::string &fname_inp, const std::string &fname_out, int itype)
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

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp)
    {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout)
    {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        finp.read((char *)&magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
            return false;
        }

        fout.write((char *)&magic, sizeof(magic));
    }

    vit_hparams hparams;

    // load hparams
    {
        finp.read((char *)&hparams.hidden_size, sizeof(hparams.hidden_size));
        finp.read((char *)&hparams.num_hidden_layers, sizeof(hparams.num_hidden_layers));
        finp.read((char *)&hparams.num_attention_heads, sizeof(hparams.num_attention_heads));
        finp.read((char *)&hparams.num_classes, sizeof(hparams.num_classes));
        finp.read((char *)&hparams.patch_size, sizeof(hparams.patch_size));
        finp.read((char *)&hparams.img_size, sizeof(hparams.img_size));
        finp.read((char *)&hparams.ftype, sizeof(hparams.ftype));

        printf("%s: hidden_size            = %d\n", __func__, hparams.hidden_size);
        printf("%s: num_hidden_layers      = %d\n", __func__, hparams.num_hidden_layers);
        printf("%s: num_attention_heads    = %d\n", __func__, hparams.num_attention_heads);
        printf("%s: patch_size             = %d\n", __func__, hparams.patch_size);
        printf("%s: img_size               = %d\n", __func__, hparams.img_size);
        printf("%s: num_classes            = %d\n", __func__, hparams.num_classes);
        printf("%s: ftype                  = %d\n", __func__, hparams.ftype);
        printf("%s: itype                  = %d\n", __func__, itype);

        fout.write((char *)&hparams.hidden_size, sizeof(hparams.hidden_size));
        fout.write((char *)&hparams.num_hidden_layers, sizeof(hparams.num_hidden_layers));
        fout.write((char *)&hparams.num_attention_heads, sizeof(hparams.num_attention_heads));
        fout.write((char *)&hparams.num_classes, sizeof(hparams.num_classes));
        fout.write((char *)&hparams.patch_size, sizeof(hparams.patch_size));
        fout.write((char *)&hparams.img_size, sizeof(hparams.img_size));
        fout.write((char *)&itype, sizeof(hparams.ftype));
    }

    printf("%s: Loaded hparams \n", __func__);

    // load class map
    {
        // read id2label from finp
        int num_labels;
        finp.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));

        for (int i = 0; i < num_labels; ++i)
        {
            int key;
            int value_length;
            finp.read(reinterpret_cast<char *>(&key), sizeof(key));
            finp.read(reinterpret_cast<char *>(&value_length), sizeof(value_length));

            std::string value(value_length, '\0');
            finp.read(&value[0], value_length);

            hparams.id2label[key] = value;
        }

        // write the id2label to fout
        fout.write(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));

        for (const auto &pair : hparams.id2label)
        {
            fout.write(reinterpret_cast<const char *>(&pair.first), sizeof(pair.first));

            int value_length = pair.second.size();
            fout.write(reinterpret_cast<const char *>(&value_length), sizeof(value_length));
            fout.write(pair.second.data(), value_length);
        }
    }

    printf("%s: Loaded id2label \n", __func__);

    // load weights
    {
        size_t total_size_org = 0;
        size_t total_size_new = 0;

        std::vector<float> work;

        std::vector<uint8_t> data_u8;
        std::vector<ggml_fp16_t> data_f16;
        std::vector<float> data_f32;

        std::vector<int64_t> hist_all(1 << 4, 0);

        while (true)
        {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            finp.read(reinterpret_cast<char *>(&length), sizeof(length));
            finp.read(reinterpret_cast<char *>(&ftype), sizeof(ftype));

            if (finp.eof())
            {
                break;
            }

            // int32_t nelements = 1;
            // int32_t ne[2] = {1, 1};
            int32_t nelements = 1;
            int32_t ne[4] = {1, 1, 1, 1};

            for (int i = 0; i < n_dims; ++i)
            {
                finp.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            finp.read(&name[0], length);

            {
                static const char *ftype_str[] = {
                    "f32",
                    "f16",
                    "q4_0",
                    "q4_1",
                };
                printf("%48s - [%5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ftype_str[ftype]);
            }

            // regexes of tensor names to be quantized
            const std::vector<std::string> k_names = {
                ".*weight",
            };

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
            quantize &= (n_dims == 2);

            if (quantize)
            {
                if (ftype == 1)
                {
                    data_f16.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                    data_f32.resize(nelements);
                    for (int i = 0; i < nelements; ++i)
                    {
                        data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                    }
                }
                else
                {
                    data_f32.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
                }

                ftype = itype;
            }
            else
            {
                const int bpe = (ftype == 0) ? sizeof(float) : sizeof(uint16_t);

                data_u8.resize(nelements * bpe);
                finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
            }

            fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fout.write(reinterpret_cast<char *>(&length), sizeof(length));
            fout.write(reinterpret_cast<char *>(&ftype), sizeof(ftype));

            for (int i = 0; i < n_dims; ++i)
            {
                fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            }
            fout.write(&name[0], length);

            if (quantize)
            {
                printf("quantizing .. ");
                work.resize(nelements); // for quantization

                size_t cur_size = 0;
                std::vector<int64_t> hist_cur(1 << 4, 0);

                switch (type)
                {
                case GGML_TYPE_Q4_0:
                {
                    cur_size = ggml_quantize_q4_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                }
                break;
                case GGML_TYPE_Q4_1:
                {
                    cur_size = ggml_quantize_q4_1(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                }
                break;
                case GGML_TYPE_Q5_0:
                {
                    cur_size = ggml_quantize_q5_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                }
                break;
                case GGML_TYPE_Q5_1:
                {
                    cur_size = ggml_quantize_q5_1(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                }
                break;
                case GGML_TYPE_Q8_0:
                {
                    cur_size = ggml_quantize_q8_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                }
                break;
                default:
                {
                    fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, type);
                    return false;
                }
                }

                fout.write(reinterpret_cast<char *>(work.data()), cur_size);
                total_size_new += cur_size;

                printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float) / 1024.0 / 1024.0, cur_size / 1024.0 / 1024.0);
                for (int i = 0; i < hist_cur.size(); ++i)
                {
                    hist_all[i] += hist_cur[i];
                }

                for (int i = 0; i < hist_cur.size(); ++i)
                {
                    printf("%5.3f ", hist_cur[i] / (float)nelements);
                }
                printf("\n");
            }
            else
            {
                printf("size = %8.3f MB\n", data_u8.size() / 1024.0 / 1024.0);
                fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
                total_size_new += data_u8.size();
            }

            total_size_org += nelements * sizeof(float);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
        printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);

        {
            int64_t sum_all = 0;
            for (int i = 0; i < hist_all.size(); ++i)
            {
                sum_all += hist_all[i];
            }

            printf("%s: hist: ", __func__);
            for (int i = 0; i < hist_all.size(); ++i)
            {
                printf("%5.3f ", hist_all[i] / (float)sum_all);
            }
            printf("\n");
        }
    }

    finp.close();
    fout.close();

    return true;
}

// usage:
// ./quantize models/ggml-model-f16.gguf models/ggml-model-f16-quant.gguf 2
//

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "usage: %s model-f32.bin model-quant.bin type\n", argv[0]);
        fprintf(stderr, "  type = 2 - q4_0\n");
        fprintf(stderr, "  type = 3 - q4_1\n");
        fprintf(stderr, "  type = 6 - q5_0\n");
        fprintf(stderr, "  type = 7 - q5_1\n");
        fprintf(stderr, "  type = 8 - q8_0\n");
        return 1;
    }

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = {0, NULL};
        struct ggml_context *ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const int itype = atoi(argv[3]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!vit_model_quantize(fname_inp, fname_out, itype))
        {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us / 1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    return 0;
}