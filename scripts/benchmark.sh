#!/bin/bash

# arrays
declare -a models=("tiny" "small" "base" "large")
declare -a quant_names=("q4_0" "q4_1" "q5_0" "q5_1" "q8_0")
declare -a quant_ids=(2 3 6 7 8)
# associative array
declare -A speed_results
declare -A memory_results

# defaults
num_threads=4
quantize_flag=0 # 0 for no quantization, 1 for quantization
N=100 # number of times to run each model

if [ "$#" -ge 1 ]; then
    echo "num_threads=$1"
    num_threads=$1
fi

if [ "$#" -ge 2 ]; then
    echo "quantize_flag=$2"
    quantize_flag=$2
fi


for model in "${models[@]}"; do
    # convert the model to gguf
    echo "Converting model: vit_${model}_patch16_224.augreg_in21k_ft_in1k"
    python convert-pth-to-ggml.py --model_name "vit_${model}_patch16_224.augreg_in21k_ft_in1k" --ftype 1 > /dev/null 2>&1

    cd build/ || exit

    # quantize the model
    if [ "$quantize_flag" -eq 1 ]; then
        for i in "${!quant_ids[@]}"; do
            q="${quant_names[$i]}"
            q_index="${quant_ids[$i]}"
            echo "Quantizing ... to ${q} ie ${q_index}"
            ./bin/quantize ../ggml-model-f16.gguf ../ggml-model-f16-quant.gguf ${q_index} > /dev/null 2>&1
            
            sum=0
            mem_usage=0

            for ((i=1; i<=N; i++)); do
                start=$(date +%s%N)
                /usr/bin/time -f "%M" -o mem.txt ./bin/vit -t $num_threads -m ../ggml-model-f16-quant.gguf -i ../assets/tench.jpg > /dev/null 2>&1
                end=$(date +%s%N)
                diff=$((end-start))
                sum=$((sum+diff))
                mem_usage=$(($mem_usage + $(cat mem.txt)))
            done

            avg_mem_usage=$(($mem_usage / N / 1024))
            avg_speed=$(($sum / N / 1000000))

            speed_results["$model,${q}"]=$avg_speed
            memory_results["$model,${q}"]=$avg_mem_usage

            rm mem.txt
        done
    else
        echo "No quantization ... for model $model"
        # run N times
        sum=0
        mem_usage=0

        for ((i=1; i<=N; i++)); do
            start=$(date +%s%N)
            /usr/bin/time -f "%M" -o mem.txt ./bin/vit -t $num_threads -m ../ggml-model-f16.gguf -i ../assets/tench.jpg > /dev/null 2>&1
            end=$(date +%s%N)
            diff=$((end-start))
            sum=$((sum+diff))
            mem_usage=$(($mem_usage + $(cat mem.txt)))
        done

        avg_mem_usage=$(($mem_usage / N / 1024))
        avg_speed=$(($sum / N / 1000000))

        speed_results["$model"]=$avg_speed
        memory_results["$model"]=$avg_mem_usage

        rm mem.txt
    fi
    
    cd ..

done

# kind of a poor man's table
if [ "$quantize_flag" -eq 1 ]; then
    echo "| Model  | Quantization | Speed (ms)    | Mem (MB)          |"
    echo "| :----: | :----------: | :-----------: | :---------------: |"

    for model in "${models[@]}"; do
        for i in "${!quant_ids[@]}"; do
            quant_name="${quant_names[$i]}"
            key="$model,$quant_name"
            if [ -v speed_results[$key] ]; then
                echo "|   $model |     $quant_name     |    ${speed_results[$key]} ms     |   ${memory_results[$key]} MB        |"
            fi
        done
    done
else
    echo "| Model  | Speed (ms)    | Mem (MB)          |"
    echo "| :----: | :-----------: | :---------------: |"

    for model in "${models[@]}"; do
        key="$model"
        echo "|   $model |    ${speed_results[$key]} ms     |   ${memory_results[$key]} MB        |"
    done
fi