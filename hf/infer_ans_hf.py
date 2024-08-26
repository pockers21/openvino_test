import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="加载和使用语言模型")
parser.add_argument("--use_cache", default=False, help="use cache or not", action='store_true')
parser.add_argument("--do_sample", default=False, help="do sample or not", action='store_true')
parser.add_argument("--device", type=str, default="cpu", help="device name")
parser.add_argument("--model", type=str, default="gpt2", help="model name or path")

# 解析命令行参数
args = parser.parse_args()
print(args)

# 加载模型和分词器
device = torch.device(args.device)
model = AutoModelForCausalLM.from_pretrained(args.model)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model)

# 定义生成文本的函数
def predict(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(inputs["input_ids"],
                            max_new_tokens=50,
                            do_sample=args.do_sample,
                            temperature=0.7,
                            eos_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(output[0], skip_special_tokens=True)
    print(resp)

# 测试生成
prompt = '写一首七言绝句'
predict(prompt)
