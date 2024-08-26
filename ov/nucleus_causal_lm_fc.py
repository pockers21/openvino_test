#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import time
import argparse
import openvino_genai
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser(description="加载和使用语言模型")
parser.add_argument("--device", type=str, default="CPU", help="device name")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompts', nargs='+')
    parser.add_argument('device')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    system = '请根据问题决定是否使用工具，若使用工具则并根据问题给出正确的函数调用- [{"description":"播放某部影片","name":"a_playOneMovie","parameters":{"properties":{"name":{"description":">影片名称","type":"string"}},"required":[],"type":"object"}}]'
    prompt = '播放电影卧虎藏龙'
    conv = [{"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")

    device = args.device  # GPU can be used as well
    print(f'use {device}')

    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    config.do_sample=True      
    config.top_p=0.8             
    config.top_k=20               
    config.temperature=0.7   

    print(f'inputs:{inputs}')
    inputs_list = inputs[0].tolist()
    decoded_text = tokenizer.decode(inputs_list)

    print("====")
    # 输出解码后的文本
    #print('Decoded text:', decoded_text)
    print("=====")


    start_time = time.time() 
    res = pipe.generate(decoded_text, config)
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print(f'cost:{elapsed_time}')
    print(f'res:{res}')


if '__main__' == __name__:
    main()

