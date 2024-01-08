#include "vit.h"
#include "ggml/ggml.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace std;

vector<string> read_class_names(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Cannot open file: " << filename << endl;
        return {};
    }

    json j;
    file >> j;

    vector<string> classNames = j.get<vector<string>>();

    return classNames;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: " << argv[0] << " <model_path> <dataset_dir> <num_images_per_class> [output_file]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string dataset_dir = argv[2];
    int num_images_per_class = std::stoi(argv[3]);
    std::string output_file = (argc == 5) ? argv[4] : "predictions.txt";

    vit_state state;
    vit_params params;
    vit_model model;
    std::vector<std::pair<float, int>> predictions;

    // read class names from JSON
    fs::path classnames_path = fs::path(dataset_dir).parent_path() / "classnames.json";
    std::string classnames_file_path = classnames_path.string();
    vector<string> CLASS_NAMES = read_class_names(classnames_path);

    if (!vit_model_load(model_path, model))
    {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return 1;
    }

    std::ofstream out_file(output_file);
    if (!out_file)
    {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return 1;
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
    }

    int total_images = 0;
    int correct_predictions = 0;

    for (const auto &class_entry : fs::directory_iterator(dataset_dir))
    {
        if (!class_entry.is_directory())
            continue;

        std::string class_name = class_entry.path().filename().string();
        int images_processed = 0;

        for (const auto &image_entry : fs::directory_iterator(class_entry.path()))
        {
            // if (images_processed++ >= num_images_per_class)
            //     break;

            // all images are in JPEG
            if (image_entry.path().extension() != ".JPEG")
                continue;

            std::string image_path = image_entry.path().string();

            // image loading and preprocessing
            image_u8 img;
            if (!load_image_from_file(image_path, img))
            {
                std::cerr << "Failed to load image from " << image_path << std::endl;
                continue;
            }

            image_f32 processed_img;
            if (!vit_image_preprocess(img, processed_img, model.hparams))
            {
                std::cerr << "Error in preprocessing image " << image_path << std::endl;
                continue;
            }

            // inference
            if (vit_predict(model, state, processed_img, params, predictions) != 0)
            {
                std::cerr << "Inference failed for image " << image_path << std::endl;
                continue;
            }

            int top_prediction_index = predictions.front().second;
            std::string class_name_prediction = CLASS_NAMES[top_prediction_index];
            if (class_name == class_name_prediction)
            {
                ++correct_predictions;
            }
            ++total_images;

            // write predictions to the output file
            out_file << image_entry.path().filename().string() << "," << class_name << "," << class_name_prediction;
            out_file << std::endl;
        }
    }

    // calculate accuracy
    double accuracy = static_cast<double>(correct_predictions) / total_images;
    std::cout << "Top-1 Accuracy: " << accuracy * 100.0 << "%" << std::endl;

    ggml_free(model.ctx);
    out_file.close();

    return 0;
}