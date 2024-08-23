import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time

# 设置命令行参数解析
parser = argparse.ArgumentParser(description="加载和使用语言模型")
parser.add_argument("--do_sample", default=False, help="do sample or not", action='store_true')
parser.add_argument("--device", type=str, default="cpu", help="device name")
parser.add_argument("--model", type=str, default="gpt2", help="model name or path")
args = parser.parse_args()

# 打印命令行参数
print(args)

# 加载模型和分词器
model_name = args.model
device = torch.device(args.device)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 将模型设置为评估模式
model.eval()

# 系统提示信息
system = '请根据问题决定是否使用工具，若使用工具则并根据问题给出正确的函数调用- [{"description":"播放某部影片","name":"a_playOneMovie","parameters":{"properties":{"name":{"description":"影片名称","type":"string"}},"required":[],"type":"object"}}]'
prompt = '播放电影卧虎藏龙'

# 预测函数
def predict(prompt: str):
    conv = [{"role": "system", "content": system},
            {"role": "user", "content": prompt}]
    
    # 使用模板将对话转换为输入格式
    input_text = system + " " + prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # 记录开始时间
    start_time = time.time() 
    
    # 生成文本
    output = model.generate(
        inputs['input_ids'],
        max_length=inputs['input_ids'].shape[1] + 100,
        do_sample=args.do_sample,
        top_p=0.8,
        top_k=20,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 记录结束时间
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print(f'cost: {elapsed_time} seconds')
    
    # 解码生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

# 调用预测函数
predict(prompt)
