import json
import re
import math_verify

def calculate_score(json_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_score = 0
    
    for item in data:
        # 解析正确答案
        correct_answer_str = item['answer']
        correct_answer = math_verify.parse(correct_answer_str)
        
        # 检查每个生成结果
        found_correct = False
        for generation in item['generations']:
            # 使用正则表达式匹配所有<answer>标签内容
            #matches = re.findall(r'\$\\boxed{(.*?)}\$', generation, re.DOTALL)
            matches = re.findall(r'<answer>(.*?)</answer>', generation, re.DOTALL)
            if not matches:
                continue  # 没有找到答案标签，跳过
            
            # 取最后一个答案
            model_answer_str = matches[-1].strip()
            
            # 解析模型答案
            try:
                model_answer = math_verify.parse(model_answer_str)
            except:
                continue  # 解析失败，跳过
            
            # 验证答案是否正确
            if math_verify.verify(model_answer, correct_answer):
                found_correct = True
                break  # 找到一个正确即可
        
        if found_correct:
            total_score += 1
    
    return total_score, len(data), total_score / len(data)

# 示例使用
if __name__ == "__main__":
    score, total, ratio = calculate_score('validation_results_step_220_small_0.7_1000.json')
    print(f"模型的总得分为: {ratio:.3f} ({score} / {total})")

    # 3B: 0.6
    # 1.5B-RL: 0.46
    # 1.5B: 0.1