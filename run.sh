set -e
set -x
mode="ans"
declare -A normal_exe_script_dict


normal_exe_script_dict["ans"]="infer_asn.py"
normal_exe_script_dict["fc"]="infer_fc.py"


base_data_dir="/home/uos/openvino_test/openvino_model_7b/"
model_dirs=(
    uos_7b_cache_false_dir=${base_data_dir}uos_july_7b_818_cache_false
    uos_7b_cache_true_dir=${base_data_dir}uos_july_7b_818_cache_true
    qwen2_7b_instruct_cache_false_dir=${base_data_dir}qwen2_7b_instruct_818_cache_false
    qwen2_7b_instruct_cache_true_dir=${base_data_dir}qwen2_7b_instruct_818_cache_true
    ）


cd ${base_code_dir}
script=${normal_exe_script_dict[$mode]}
for model_dir in "${model_dirs[@]}"; do
    # 检查缓存情况
    if [[ "$model_dir" == *"cache_false" ]]; then
        #qwen2 7b
        #cache false
        python ${script} --do_sample --device="GPU"  --model $model_dir 
        python ${script} --do_sample --device="CPU"  --model $model_dir
        python ${script}  --device="GPU"     --model $model_dir
        python ${script}  --device="CPU"     --model $model_dir
    else
        python ${script} --use_cache     --do_sample     --device="GPU"  --model $model_dir
        python ${script} --use_cache     --do_sample     --device="CPU"  --model $model_dir
        python ${script}  --use_cache    --device="GPU"     --model $model_dir
        python ${script}  --use_cache    --device="CPU"     --model $model_dir
    fi
done




base_genai_code_dir="/home/uos/openvino_test/openvino.genai/samples/python/"
greedy_dir=${base_code_dir}"greedy_causal_lm/"
beam_dir=${base_code_dir}"beam_search_causal_lm/"

declare -A genai_exe_script_dict

for model_dir in "${model_dirs[@]}"; do
    cd greedy_dir
    #cache false
    if [[ "$mode" == "ans" ]]; then
        script="greedy_causal_lm.py"
    else
        script="greedy_causal_lm_fc.py"
    fi
    python ${script} $model_dir "Why is the Sun yellow?" CPU
    python ${script} $model_dir "Why is the Sun yellow?" GPU
    
    cd beam_dir
    if [[ "$mode" == "ans" ]]; then
        script="beam_search_causal_lm.py"
    else
        script="beam_search_causal_lm_fc.py"
    fi
    python ${script} $model_dir "Why is the Sun yellow?" CPU
    python ${script} $model_dir "Why is the Sun yellow?" GPU
        
    
done









