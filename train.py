import os
import gc
import json
import re
import sys
sys.float_info.dig = 4

from question_iterator import QuestionIterator

from trl.trainer.utils import selective_log_softmax
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import logsigmoid
import math_verify
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    AutoConfig,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig
from torch.optim import AdamW
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import bitsandbytes as bnb

# sudo nvidia-smi -i 0 -pl 320

# 加载原始模型
model_dir = "/home/leo/NLP/models/Qwen-VL/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)

# 添加LoRA适配器
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules=["gate_proj", "up_proj", "down_proj", "lm_head"],
    modules_to_save=["embed_tokens", "norm"],
)
lora_model = get_peft_model(model, peft_config)
lora_model.print_trainable_parameters()
lora_model.gradient_checkpointing = True

# 优化器设置
# optimizer = AdamW(lora_model.parameters(), lr=1e-5)
optimizer = bnb.optim.AdamW8bit(lora_model.parameters(), lr=1e-5)

# 数据加载
# def load_jsonl_data(file_path):
#     data = []
#     with open(file_path, "r") as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data
#
#dataset = load_jsonl_data("./datasets/OpenR1-Math-220k/train-00000-of-00010.jsonl")
with open("question_in_levels.json", "r", encoding="utf8") as f:
    data = json.load(f)

dataset = QuestionIterator(
    data_dict=data,
    max_epochs=2,
    sample_counts={"easy": 2, "medium": 2, "hard":1},
    base_category="medium"
)


# 生成配置
start_to_punish = 800
max_new_tokens = 1000

gen_config = GenerationConfig.from_pretrained(model_dir)
gen_config.max_new_tokens = max_new_tokens
gen_config.temperature = 0.7
gen_config.do_sample = True
gen_config.num_return_sequences = 20
gen_config.use_cache = True


# RL
def get_log_softmax_on_out(model, original_inputs: dict, outputs: str):
    inputs = tokenizer([outputs], return_tensors="pt", max_length=1500, truncation=True).to(model.device)
    inputs = lora_model.prepare_inputs_for_generation(**inputs)
    output = model(**inputs) # 为节省内存，重算一遍
    shifted_logits = output.logits[:, :-1, :]
    shifted_ids = inputs["input_ids"][:, 1:]
    log_softmax = selective_log_softmax(shifted_logits, shifted_ids)
    original_length = len(original_inputs["input_ids"][0])
    return log_softmax[0, original_length+1:]

def loss_grdpo(model, inputs: dict, good: str, bad: str, step_size, beta):
    good_prob = get_log_softmax_on_out(model, inputs, good).mean()
    bad_prob = get_log_softmax_on_out(model, inputs, bad).mean()
    logsig = logsigmoid(beta * (good_prob - bad_prob))
    return (-logsig) / step_size

# 辅助函数
def form_prompt(item):
    """构造问题提示模板"""
    problem = item["problem"]
    return f"""Please solve the following math problem:
In the end, provide the final answer, wrapping the answer in the answer tag.
Example: `The answer is: <answer>42</answer>`

Problem: {problem}
Solution process:"""

def extract_answer(response):
    match = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return match[-1] if match else None

def reward_answer_correct(answer_from_response, answer):
    answer_from_response = math_verify.parse(answer_from_response)
    answer = math_verify.parse(answer)
    return 1.0 if math_verify.verify(answer_from_response, answer) else 0.0

def reward_answer_tag_at_end(response):
    match = re.match(r".*<answer>(.*?)</answer>$", response, re.DOTALL)
    return 1.0 if match else 0.0

def response_reward(item, response, length):
    prompt_text = item["problem"]
    # 正确性奖励
    answer = extract_answer(response[len(prompt_text):])
    reward_correctness = reward_answer_correct(answer, item["answer"]) if answer else 0.0
    # 格式奖励
    #   answer tag奖励
    #     1. 有answer tag
    reward_has_tag = 1.0 if answer else 0.0
    #     2. answer tag在最后
    reward_tag_on_end = reward_answer_tag_at_end(response)
    #   长度奖励
    reduce_length = max_new_tokens - start_to_punish
    reward_length = min(max(0.0, (reduce_length - (length - start_to_punish)) / reduce_length ), 1.0)
    response_reward = reward_correctness + (reward_has_tag + reward_tag_on_end + reward_length) / 3
    return response_reward

def calculate_rewards(inputs, responses, item, tokenizer):
    rewards = []
    inputs_length = len(inputs["input_ids"][0])
    out_lengths = [len(tokenizer.encode(resp)) for resp in responses]
    print(f"out lengths: {out_lengths}")
    lengths = [out_len - inputs_length for out_len in out_lengths]

    for resp, length in zip(responses, lengths):
        reward = response_reward(item, resp, length)
        rewards.append(reward)
    return rewards

# 训练循环
save_per_groups = 10
question_group_size = 5
total_steps = 0
os.makedirs("outputs", exist_ok=True)

for group_idx, group_items in enumerate(dataset):
    step_id = group_idx
    optimizer.zero_grad()
    total_loss = 0.0
    
    for item in group_items:
        # 生成样本
        prompt = form_prompt(item)
        print(repr(item["problem"][:100]+"...") if len(item["problem"]) > 100 else "")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        lora_model.eval()
        with torch.no_grad():
            outputs = lora_model.generate(
                **inputs,
                generation_config=gen_config,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 计算奖励并排序
        rewards = calculate_rewards(inputs, responses, item, tokenizer)
        print(f"rewards: {[float(r) for r in rewards]}")
        sorted_pairs = sorted(zip(responses, rewards), key=lambda x: -x[1])
        sorted_responses = [x[0] for x in sorted_pairs]
        
        # 取前5和后5
        good_responses = sorted_responses[:10]
        bad_responses = sorted_responses[-10:]
        
        lora_model.train()
        # 计算损失
        for good in tqdm(good_responses):
            for bad in bad_responses:
                gc.collect()
                torch.cuda.empty_cache()
                loss = loss_grdpo(
                    lora_model,
                    inputs,
                    good=good,
                    bad=bad,
                    step_size=1,
                    beta=0.1
                )
                loss.backward()
                total_loss += float(loss)
    
    # 梯度裁剪和优化
    gc.collect()
    torch.cuda.empty_cache()
    torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # 记录日志
    avg_loss = total_loss / (len(group_items) * len(good_responses) * len(bad_responses))
    print(f"Group {step_id} Loss: {avg_loss:.4f}")
    
    # 保存模型
    if step_id % save_per_groups == 0 and group_idx != 0:
        save_path = f"outputs/step_{step_id}"
        lora_model.save_pretrained(save_path, save_embedding_layers=False)
        print(f"Model saved at {save_path}")
