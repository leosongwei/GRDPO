import os
import json
import re
import math_verify
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def load_jsonl_data(file_path):
    """加载JSONL格式的数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def form_prompt(item):
    """构造问题提示模板"""
    problem = item["problem"]
    return f"""Please solve the following math problem:
In the end, provide the final answer, wrapping the answer in the answer tag.
Example: `The answer is: <answer>42</answer>`

Problem: {problem}
Solution process:"""

def extract_answer(response):
    """从模型响应中提取答案"""
    matches = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
    # match = re.search(r'\$\\boxed{(.*?)}\$', response, re.DOTALL)
    return matches[-1] if matches else None

# 初始化模型和tokenizer
#model_dir = "/home/leo/NLP/models/Qwen-VL/Qwen2.5-1.5B-Instruct"
model_dir = "./outputs/step_220_small"
original_model_dir = "/home/leo/NLP/models/Qwen-VL/Qwen2.5-1.5B-Instruct"
#original_model_dir = model_dir

model_name = model_dir.split("/")[-1]
#tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
gen_config = GenerationConfig.from_pretrained(original_model_dir)
tokenizer = AutoTokenizer.from_pretrained(original_model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
    #attn_implementation="flash_attention_2"
)
model.eval()

# 配置生成参数

#gen_config = GenerationConfig.from_pretrained(model_dir)
gen_config.max_new_tokens = 1000
gen_config.temperature = 0.7
gen_config.do_sample = True
gen_config.num_return_sequences = 10
gen_config.use_cache = True

# 加载数据
dataset = load_jsonl_data("./datasets/OpenR1-Math-220k/train-00000-of-00010.jsonl")

# 处理每个题目
results = []

# for item in tqdm(dataset[:519]):
for item in tqdm(dataset[1000:1100]):
    prompt = form_prompt(item)
    
    # 生成响应
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)
    
    # 解码和验证
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    item["generations"] = responses
    print("Problem:", item["problem"])
    print("Answer", item["answer"])
    print("out-1:", responses[-1])
    print("try extract answer:", extract_answer(responses[-1]))
    print("ground truth:", item["answer"])
    print("======================================================================================")
    results.append(item)
    print()
    # 实时保存结果
    with open(f"validation_results_{model_name}_0.7_1000.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print("处理完成，结果已保存到enhanced_results.json")