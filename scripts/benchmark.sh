#!/bin/bash

# arrays
declare -a models=("tiny" "small" "base" "large")
declare -a speed_results
declare -a memory_results

for model in "${models[@]}"; do
    echo "Converting model: vit_${model}_patch16_224.augreg_in21k_ft_in1k"
    python convert-pth-to-ggml.py --model_name "vit_${model}_patch16_224.augreg_in21k_ft_in1k" --ftype 1 > /dev/null 2>&1

    cd build/ || exit

    echo "Quantizing ..."
    ./bin/quantize ../ggml-model-f16.gguf ../ggml-model-f16-quant.gguf 2 > /dev/null 2>&1

    # run N times
    N=10
    sum=0
    mem_usage=0

    for ((i=1; i<=N; i++)); do
        start=$(date +%s%N)
        /usr/bin/time -f "%M" -o mem.txt ./bin/vit -t 4 -m ../ggml-model-f16-quant.gguf -i ../assets/tench.jpg > /dev/null 2>&1
        end=$(date +%s%N)
        diff=$((end-start))
        sum=$((sum+diff))
        mem_usage=$(($mem_usage + $(cat mem.txt)))
    done

    avg_mem_usage=$(($mem_usage / N / 1024))
    avg_speed=$(($sum / N / 1000000))

    speed_results+=("$avg_speed")
    memory_results+=("$avg_mem_usage")

    # del the mem file / back to parent
    rm mem.txt
    cd ..
done


# kind of poor man's table
echo "| Model  | Speed (ms)    | Mem (MB)          |"
echo "| :----: | :-----------: | :---------------: |"

for i in "${!models[@]}"; do
    echo "|   ${models[$i]} |    ${speed_results[$i]} ms     |   ${memory_results[$i]} MB        |"
done