import os
import gc
import json
import re
import statistics
from typing import List

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
    AutoConfig
)
from peft import get_peft_model, LoraConfig
from torch.optim import AdamW
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import bitsandbytes as bnb
from torch.utils.tensorboard import SummaryWriter

# --------------------------------------------
# tensorboard config
train_name = "20250313_1"
# 配置
model_dir = "/home/leo/NLP/models/Qwen-VL/Qwen2.5-1.5B-Instruct"
loss_compute_group_size = 1
sample_counts = {"easy": 1, "medium": 2, "hard":2} # 每步题目配比
save_per_groups = 2
max_epochs = 10

# 优化器
lr=1e-5

# LoRA
lora_rank = 16
lora_alpha = 32

# RL
## 长度惩罚参数
start_to_punish = 800
max_new_tokens = 1000

# 生成配置
gen_config = GenerationConfig.from_pretrained(model_dir)
gen_config.max_new_tokens = max_new_tokens
gen_config.temperature = 0.7
gen_config.do_sample = True
gen_config.num_return_sequences = 20
gen_config.use_cache = True
# ---------------------------------------------

log_writer = SummaryWriter(
    log_dir="./runs",
    flush_secs=1,
    comment=train_name
)

# 加载原始模型

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
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=["gate_proj", "up_proj", "down_proj", "lm_head"],
    modules_to_save=["embed_tokens", "norm"],
)
lora_model = get_peft_model(model, peft_config)
lora_model.print_trainable_parameters()
lora_model.gradient_checkpointing = True

# 优化器设置
optimizer = bnb.optim.AdamW8bit(lora_model.parameters(), lr=lr)

# 数据加载
with open("question_in_levels.json", "r", encoding="utf8") as f:
    data = json.load(f)

dataset = QuestionIterator(
    data_dict=data,
    max_epochs=max_epochs,
    sample_counts=sample_counts,
    base_category="medium"
)

# RL
def get_log_softmax_mean_on_out(model, original_inputs: dict, outputs: str):
    if isinstance(outputs, str):
        outputs = [outputs]
    inputs = tokenizer(outputs, return_tensors="pt", padding=True, max_length=1500, truncation=True).to(model.device)
    inputs = model.prepare_inputs_for_generation(**inputs)
    output = model(**inputs) # 为节省内存，重算一遍
    shifted_logits = output.logits[:, :-1, :]
    shifted_ids = inputs["input_ids"][:, 1:]
    log_softmax = selective_log_softmax(shifted_logits, shifted_ids)

    original_len = original_inputs["attention_mask"].shape[1]
    shifted_mask = inputs["attention_mask"][:, 1:] # follow shifted_ids
    bz, max_output_length = inputs["attention_mask"].shape
    shifted_pos = torch.arange(0, max_output_length, device=model.device)[1:].repeat(bz, 1)
    shifted_mask = torch.where(shifted_pos < original_len, 0, shifted_mask)
    outs = (log_softmax * shifted_mask).sum(dim=-1) / (shifted_mask.sum(dim=-1))
    return outs

def loss_grdpo(model, inputs: dict, good: str, bads: List[str], beta):
    good_prob = -get_log_softmax_mean_on_out(model, inputs, good)[0]
    bad_prob = -get_log_softmax_mean_on_out(model, inputs, bads)
    logsig = logsigmoid(beta * (good_prob - bad_prob))
    return -logsig

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
    
    lengths = [out_len - inputs_length for out_len in out_lengths]

    for resp, length in zip(responses, lengths):
        reward = response_reward(item, resp, length)
        rewards.append(reward)
    return rewards, lengths


# 训练循环
os.makedirs("outputs", exist_ok=True)

for group_idx, group_items in enumerate(dataset):
    step_id = group_idx
    optimizer.zero_grad()

    total_loss = 0.0
    all_rewards = []
    all_lengths = []
    
    for item in group_items:
        # 生成样本
        prompt = form_prompt(item)
        print(repr(item["problem"][:100]+"...") if len(item["problem"]) > 100 else "")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        gc.collect()
        torch.cuda.empty_cache()
        lora_model.eval()
        with torch.no_grad():
            outputs = lora_model.generate(
                **inputs,
                generation_config=gen_config,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 计算奖励并排序
        rewards, lengths = calculate_rewards(inputs, responses, item, tokenizer)
        all_rewards += rewards
        all_lengths += lengths
        print(f"rewards: {[f"{float(r):.3f}" for r in rewards]}")
        print(f"out lengths: {lengths}")
        sorted_pairs = sorted(zip(responses, rewards), key=lambda x: -x[1])
        sorted_responses = [x[0] for x in sorted_pairs]
        
        # 取前5和后5
        good_responses = sorted_responses[:10]
        bad_responses = sorted_responses[-10:]
        
        lora_model.train()
        # 计算损失
        for good in tqdm(good_responses):
            for i in range(0, len(bad_responses), loss_compute_group_size):
                bads_group = bad_responses[i:i+loss_compute_group_size]
                gc.collect()
                torch.cuda.empty_cache()
                loss = loss_grdpo(
                    lora_model, inputs,
                    good=good,
                    bads=bads_group,
                    beta=0.1
                ).sum()
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
    avg_reward = float(statistics.mean(all_rewards))
    print(f"Group avg reward: {avg_reward:.3f}")
    avg_length = float(statistics.mean(all_lengths))
    print(f"Group avg length: {avg_length:.3f}")
    log_writer.add_scalar("avg_loss", avg_loss)
    log_writer.add_scalar("avg_reward", avg_reward)
    log_writer.add_scalar("avg_length", avg_length)
    
    # 保存模型
    if step_id % save_per_groups == 0 and group_idx != 0:
        save_path = f"outputs/step_{step_id}"
        lora_model.save_pretrained(save_path, save_embedding_layers=False)
        print(f"Model saved at {save_path}")

    print('-----------------------------------')