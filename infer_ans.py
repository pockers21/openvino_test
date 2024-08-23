import torch
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor, MinLengthLogitsProcessor
import openvino as ov
import sys
import argparse
import time


parser = argparse.ArgumentParser(description="加载和使用语言模型")
parser.add_argument("--use_cache",  default=False, help="use cache or not", action='store_true')
parser.add_argument("--do_sample",  default=False, help="do sample or not", action='store_true')
parser.add_argument("--device", type=str, default="CPU", help="device name")
parser.add_argument("--model", type=str, default="", help="dataset name")

# 解析命令行参数
args = parser.parse_args()
print(args)

core = ov.Core()
available_devices = core.available_devices
print(available_devices)

model_id=args.model
# model_id_7b = './openvino-7b'model = OVModelForCausalLM.from_pretrained(model_id, compile=False, use_cache=True, device="GPU")
model = OVModelForCausalLM.from_pretrained(model_id, compile=False, use_cache=args.use_cache, device=args.device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = '写一首七言绝句'

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# output = pipe(prompt)
# print(output)

def predict(prompt: str):
    conv = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(conv, add_generation_prompt=False, return_tensors="pt")
    start_time = time.time() 
    output = model.generate(inputs.to(model.device),
                            max_new_tokens=500, do_sample=args.do_sample, temperature=0.7,
                                eos_token_id=tokenizer.eos_token_id)
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print(f'cost:{elapsed_time}')
    resp = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print(resp)

predict(prompt)
