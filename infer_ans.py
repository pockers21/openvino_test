import torch
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor, MinLengthLogitsProcessor
import openvino as ov
import sys

core = ov.Core()
available_devices = core.available_devices
print(available_devices)

#model_id = './qwen2_7b_7_31_cache_false/qwen2_7b_7_31_cache_false'
model_id = './qwen2_7b_7_31_cache_true/qwen2_7b_7_31'
# model_id_7b = './openvino-7b'model = OVModelForCausalLM.from_pretrained(model_id, compile=False, use_cache=True, device="GPU")
model = OVModelForCausalLM.from_pretrained(model_id, compile=False, use_cache=True, device="GPU")
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

    output = model.generate(inputs.to(model.device),
                            max_new_tokens=500, do_sample=False, temperature=0.7,
                                eos_token_id=tokenizer.eos_token_id)
    resp = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print(resp)

predict(prompt)

