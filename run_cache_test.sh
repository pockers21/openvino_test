#!/bin/bash
set -x
RED='\033[0;31m'
NC='\033[0m' # 

export PS4="${RED}+\$ ${NC}"

mode="ans"

# Control variables
run_normal=true
run_genai=true

declare -A normal_exe_script_dict
normal_exe_script_dict["ans"]="infer_ans.py"
normal_exe_script_dict["fc"]="infer_fc.py"

base_data_dir="/home/uos/openvino_test/script/"

model_dirs=(
    ${base_data_dir}qwen2_1.5b_instruct_818_cache_true
    #${base_data_dir}qwen2_1.5b_instruct_818_cache_false
    #${base_data_dir}uos_july_7b_818_cache_false
    #${base_data_dir}uos_july_7b_818_cache_true
    #${base_data_dir}qwen2_7b_instruct_818_cache_false
    #${base_data_dir}qwen2_7b_instruct_818_cache_true
)

base_genai_code_dir="/home/uos/openvino_test/openvino.genai/samples/python/"
greedy_dir=${base_genai_code_dir}greedy_causal_lm
beam_dir=${base_genai_code_dir}beam_search_causal_lm

declare -A genai_exe_script_dict

run_normal_exe_script() {
    cd ${base_data_dir}
    script=${normal_exe_script_dict[$mode]}
    echo ${script}

    for model_dir in "${model_dirs[@]}"; do
        # 检查缓存情况
        if [[ "$model_dir" == *"cache_false" ]]; then
            # qwen2 7b
            # cache false
            python ${script} --do_sample --device="GPU" --model $model_dir
            python ${script} --do_sample --device="CPU" --model $model_dir
            python ${script} --device="GPU" --model $model_dir
            python ${script} --device="CPU" --model $model_dir
        else
            python ${script} --use_cache --do_sample --device="GPU" --model $model_dir
            python ${script} --use_cache --do_sample --device="CPU" --model $model_dir
            python ${script} --use_cache --device="GPU" --model $model_dir
            python ${script} --use_cache --device="CPU" --model $model_dir
        fi
    done
}

run_genai_exe_script() {
    for model_dir in "${model_dirs[@]}"; do
        #cd ${greedy_dir}
        # cache false
        if [[ "$mode" == "ans" ]]; then
            script="greedy_causal_lm.py"
        else
            script="greedy_causal_lm_fc.py"
        fi
        python ${script} $model_dir "Why is the Sun yellow?" CPU
        python ${script} $model_dir "Why is the Sun yellow?" GPU

        #cd ${beam_dir}
        if [[ "$mode" == "ans" ]]; then
            script="beam_search_causal_lm.py"
        else
            script="beam_search_causal_lm_fc.py"
        fi
        python ${script} $model_dir "Why is the Sun yellow?" CPU
        python ${script} $model_dir "Why is the Sun yellow?" GPU
    done
}

# Conditionally call the functions based on control variables
if $run_normal; then
    run_normal_exe_script
fi

if $run_genai; then
    run_genai_exe_script
fi
