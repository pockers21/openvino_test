import torch
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor, MinLengthLogitsProcessor
import openvino as ov
import sys
import argparse

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
model_id = args.model


#model_id = './qwen2_7b_7_31_cache_true/qwen2_7b_7_31'
# model_id_7b = './openvino-7b'model = OVModelForCausalLM.from_pretrained(model_id, compile=False, use_cache=True, device="GPU")
model = OVModelForCausalLM.from_pretrained(model_id, compile=True, use_cache=args.use_cache, device=args.device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
logits_processor = LogitsProcessorList()
#logits_processor.append(MinLengthLogitsProcessor(15, eos_token_id=tokenizer.eos_token))
#logits_processor.append(InfNanRemoveLogitsProcessor())


system = '你是一个DSL语法专家，你熟悉antlr设计的一种名为querylang的语言，这种语言用于描述使用各种文件属性来查询操作系统中的文件或文件内容的操作'
prompt = '我想要找那些路径跟“素材.docx”不一样的文件。'

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# output = pipe(prompt)
# print(output)

def predict(prompt: str):
    conv = [{"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")

    output = model.generate(inputs.to(model.device),
                            logits_processor=logits_processor,
                            max_new_tokens=500, do_sample=args.do_sample, temperature=0.7,
                                eos_token_id=tokenizer.eos_token_id)
    resp = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print(resp)

predict(prompt)
~                     
