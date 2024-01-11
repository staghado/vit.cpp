To run the benchmark on ImageNet1k, you first need to have the dataset downloaded in a folder `dataset`.

For this benchmark the [JSON for Modern C++ Library](https://github.com/nlohmann/json) is required to load JSON files in C++. Note that this library is not needed for other executables : `vit` and `quantize`.

## Build

    cmake -BUILD_BENCHMARK=ON ..