#!/bin/bash
set -x
RED='\033[0;31m'
NC='\033[0m' # 

export PS4="${RED}+\$ ${NC}"

mode="dsl"

# Control variables
run_normal=true
run_genai=true

declare -A normal_exe_script_dict
normal_exe_script_dict["ans"]="infer_ans_ov.py"
normal_exe_script_dict["fc"]="infer_fc_ov.py"
normal_exe_script_dict["dsl"]="infer_dsl_ov.py"

base_data_dir="/home/uos/openvino_test/openvino_test/"

model_dirs=(
    #${base_data_dir}qwen2_1.5b_v3_full_plus_ov_i8
    #${base_data_dir}qwen2_1.5B_v2_total_plus_ov_i8
    ${base_data_dir}qwen2_1.5B_v1_total_plus_ov_i8
    #${base_data_dir}qwen2_7B_v2_total_plus_ov_i8
    #${base_data_dir}qwen2_1.5b_v3_full_plus_ov
    #${base_data_dir}qwen2_7b_instruct_818_cache_true
    #{base_data_dir}qwen2_7B_v2_total_plus_ov
    #${base_data_dir}qwen2_1.5b_instruct_818_cache_true
    #${base_data_dir}qwen2_1.5b_instruct_818_cache_false
    #${base_data_dir}uos_july_7b_818_cache_false
    #${base_data_dir}uos_july_7b_818_cache_true
    #${base_data_dir}qwen2_7b_instruct_818_cache_false
    #${base_data_dir}qwen2_7b_instruct_818_cache_true
)



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

declare -A greedy_script_dict
greedy_script_dict["ans"]="greedy_causal_lm.py"
greedy_script_dict["fc"]="greedy_causal_lm_fc.py"
greedy_script_dict["dsl"]="greedy_causal_lm_dsl.py"

declare -A nucleus_script_dict
nucleus_script_dict["ans"]="nucleus_causal_lm.py"
nucleus_script_dict["fc"]="nucleus_causal_lm_fc.py"
nucleus_script_dict["dsl"]="nucleus_causal_lm_dsl.py"

run_genai_exe_script() {
    for model_dir in "${model_dirs[@]}"; do
        #cd ${greedy_dir}
        # cache false
        script=${greedy_script_dict[$mode]}

        python ${script} $model_dir "Why is the Sun yellow?" CPU
        python ${script} $model_dir "Why is the Sun yellow?" GPU

        script=${nucleus_script_dict[$mode]}
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

