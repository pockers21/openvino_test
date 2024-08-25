#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import argparse
import time

parser = argparse.ArgumentParser(description="加载和使用语言模型")
parser.add_argument("--device", type=str, default="CPU", help="device name")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    parser.add_argument('device')
    args = parser.parse_args()

    device = args.device  # GPU can be used as well
    print(f'use {device}')
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    config.do_sample=True      
    config.top_p=0.8             
    config.top_k=20               
    config.temperature=0.7        
    start_time = time.time() 
    res = pipe.generate(args.prompt, config)
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print(f'cost:{elapsed_time}')
    print(f'res:{res}')
    
    


if '__main__' == __name__:
    main()
