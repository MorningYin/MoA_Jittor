#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学问题提示词使用示例

这个文件展示了如何使用为带思维链的数学问题设计的提示词模板。
"""

import json
from MathDataset import MathChainOfThoughtDataset, PROMPT_DICT

def demonstrate_math_prompts():
    """演示不同的数学提示词模板"""
    
    # 示例数学问题
    sample_question = "Joan found 70 seashells on the beach. She gave Sam some of her seashells. She has 27 seashells. How many seashells did she give to Sam?"
    
    print("=== 数学问题提示词模板演示 ===\n")
    
    # 1. 基础思维链提示
    print("1. 基础思维链提示:")
    print(PROMPT_DICT["math_chain_of_thought"].format(question=sample_question))
    print("\n" + "="*50 + "\n")
    
    # 2. 详细推理提示
    print("2. 详细推理提示:")
    print(PROMPT_DICT["math_reasoning_detailed"].format(question=sample_question))
    print("\n" + "="*50 + "\n")
    
    # 3. 系统分析提示
    print("3. 系统分析提示:")
    print(PROMPT_DICT["math_analysis"].format(question=sample_question))
    print("\n" + "="*50 + "\n")
    
    # 4. 简洁思维链提示
    print("4. 简洁思维链提示:")
    print(PROMPT_DICT["math_chain_simple"].format(question=sample_question))
    print("\n" + "="*50 + "\n")
    
    # 5. 带验证的提示
    print("5. 带验证的提示:")
    print(PROMPT_DICT["math_with_verification"].format(question=sample_question))
    print("\n" + "="*50 + "\n")

def demonstrate_dataset_usage():
    """演示如何使用数学数据集"""
    
    print("=== 数学数据集使用示例 ===\n")
    
    # 模拟数据集配置
    config = {
        "data_path": "Dataset/math_commonsense/AddSub/addsub_1.json",
        "tokenizer_path": "path/to/tokenizer",
        "batch_size": 4,
        "max_tokens": 512,
        "partition": "train",
        "prompt_style": "random"  # 或指定具体样式
    }
    
    print("数据集配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n可用的提示样式:")
    styles = [
        "chain_of_thought",
        "detailed_reasoning", 
        "systematic_analysis",
        "simple_chain",
        "with_verification",
        "random"  # 随机选择
    ]
    for style in styles:
        print(f"  - {style}")
    
    print("\n使用建议:")
    print("1. 对于初学者：使用 'simple_chain' 或 'chain_of_thought'")
    print("2. 对于复杂问题：使用 'detailed_reasoning' 或 'systematic_analysis'")
    print("3. 对于需要验证：使用 'with_verification'")
    print("4. 为了增加多样性：使用 'random'")

def show_prompt_design_principles():
    """展示提示词设计原则"""
    
    print("=== 数学提示词设计原则 ===\n")
    
    principles = [
        {
            "principle": "问题理解引导",
            "description": "引导模型仔细阅读和理解问题，识别关键信息",
            "example": "First, carefully read and understand the problem. Then, identify the key information..."
        },
        {
            "principle": "步骤分解",
            "description": "将复杂问题分解为可管理的步骤",
            "example": "Break down the problem into smaller steps and solve each step logically"
        },
        {
            "principle": "推理过程展示",
            "description": "要求模型展示完整的推理过程，而不仅仅是答案",
            "example": "Show your work clearly and explain your reasoning"
        },
        {
            "principle": "答案验证",
            "description": "鼓励模型验证答案的合理性",
            "example": "Verify your answer makes sense"
        },
        {
            "principle": "角色定位",
            "description": "给模型明确的角色定位，如数学导师",
            "example": "You are a math tutor / You are an expert math problem solver"
        }
    ]
    
    for i, principle in enumerate(principles, 1):
        print(f"{i}. {principle['principle']}")
        print(f"   描述: {principle['description']}")
        print(f"   示例: {principle['example']}")
        print()

def compare_prompt_effectiveness():
    """比较不同提示词的效果"""
    
    print("=== 提示词效果比较 ===\n")
    
    comparison = {
        "chain_of_thought": {
            "优点": ["结构清晰", "易于理解", "适合初学者"],
            "缺点": ["可能过于冗长", "对简单问题可能过度复杂"],
            "适用场景": "基础数学问题，需要详细解释"
        },
        "detailed_reasoning": {
            "优点": ["步骤明确", "逻辑性强", "适合复杂问题"],
            "缺点": ["可能过于详细", "对简单问题冗余"],
            "适用场景": "复杂数学问题，需要严格推理"
        },
        "systematic_analysis": {
            "优点": ["分析全面", "结构系统", "易于检查"],
            "缺点": ["可能过于形式化", "缺乏灵活性"],
            "适用场景": "需要系统分析的问题"
        },
        "simple_chain": {
            "优点": ["简洁明了", "易于实现", "计算效率高"],
            "缺点": ["可能缺乏细节", "对复杂问题不够详细"],
            "适用场景": "简单数学问题，快速解答"
        },
        "with_verification": {
            "优点": ["包含验证步骤", "提高准确性", "培养检查习惯"],
            "缺点": ["增加计算开销", "可能过于谨慎"],
            "适用场景": "需要高准确性的问题"
        }
    }
    
    for prompt_type, analysis in comparison.items():
        print(f"{prompt_type}:")
        print(f"  优点: {', '.join(analysis['优点'])}")
        print(f"  缺点: {', '.join(analysis['缺点'])}")
        print(f"  适用场景: {analysis['适用场景']}")
        print()

if __name__ == "__main__":
    print("数学问题提示词设计演示\n")
    
    # 演示不同的提示词模板
    demonstrate_math_prompts()
    
    # 演示数据集使用
    demonstrate_dataset_usage()
    
    # 展示设计原则
    show_prompt_design_principles()
    
    # 比较效果
    compare_prompt_effectiveness()
    
    print("=== 总结 ===")
    print("为带思维链的数学问题设计的提示词模板具有以下特点：")
    print("1. 引导模型进行系统性思考")
    print("2. 要求展示完整的推理过程")
    print("3. 提供多种风格以适应不同需求")
    print("4. 包含验证和检查机制")
    print("5. 支持随机选择以增加训练多样性") 