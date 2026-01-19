import json
import argparse
from typing import Dict, List, Any, Tuple


# 常量定义（全大写）
INPUT_FILES = [
    "cachegen_merged_qasper_ans.json",
    "cachegen_merged_qasper_bitcomp.json",    
    "none_merged_qasper_bitcomp.json", 
    "none_merged_qasper_ans.json",
    "hadamard_merged_qasper_ans.json",
    "hadamard_merged_qasper_bitcomp.json",
    # "cachegen_merged_multi_news_ans.json",
    # "cachegen_merged_multi_news_bitcomp.json",
    # "none_merged_multi_news_ans.json",
    # "none_merged_multi_news_bitcomp.json",
    # "hadamard_merged_multi_news_ans.json",
    # "hadamard_merged_multi_news_bitcomp.json",
    # "cachegen_merged_gsm8k_ans.json",
    # "cachegen_merged_gsm8k_bitcomp.json",
    # "none_merged_gsm8k_ans.json",
    # "none_merged_gsm8k_bitcomp.json",
    # "hadamard_merged_gsm8k_ans.json",
    # "hadamard_merged_gsm8k_bitcomp.json",
    # "cachegen_merged_gsm8k_llama_ans.json",
    # "cachegen_merged_gsm8k_llama_bitcomp.json",
    # "none_merged_gsm8k_llama_ans.json",
    # "none_merged_gsm8k_llama_bitcomp.json",
    # "hadamard_merged_gsm8k_llama_ans.json",
    # "hadamard_merged_gsm8k_llama_bitcomp.json",
]

OUTPUT_FILE = "merged_output.json"
CONFIG_ID_FIELD = "config_id"


def create_signature(record: Dict[str, Any]) -> Tuple:
    """
    创建记录的签名，排除 config_id 和 id 字段
    
    参数:
    - record: 一条JSON记录
    
    返回:
    - 签名的元组
    """
    # 创建一个新字典，排除 config_id 和 id
    signature_dict = {k: v for k, v in record.items() 
                     if k not in [CONFIG_ID_FIELD]}
    
    # 将字典转换为可哈希的元组（对列表等需要特殊处理）
    def to_hashable(obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, to_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(to_hashable(item) for item in obj)
        else:
            return obj
    
    return to_hashable(signature_dict)


def merge_json_files(input_files: List[str], output_file: str) -> None:
    """
    合并多个JSON文件
    
    参数:
    - input_files: 输入文件路径列表
    - output_file: 输出文件路径
    """
    # 用于跟踪已见过的数据签名和对应的id
    signature_to_id: Dict[Tuple, int] = {}
    merged_data: List[Dict[str, Any]] = []
    next_id = 0
    
    print(f"开始合并 {len(input_files)} 个JSON文件...")
    
    # 遍历所有输入文件
    for file_path in input_files:
        print(f"  处理文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"  警告: {file_path} 不是JSON数组，跳过")
                continue
            
            # 处理每条记录
            for record in data:
                if not isinstance(record, dict):
                    continue
                
                # 创建签名
                signature = create_signature(record)
                
                # 检查是否已存在
                if signature in signature_to_id:
                    # 数据已存在，跳过
                    existing_id = signature_to_id[signature]
                    print(f"    跳过重复数据 (id={existing_id}, config_id={record.get(CONFIG_ID_FIELD, 'N/A')})")
                else:
                    # 新数据，分配新id
                    new_record = record.copy()
                    new_record[CONFIG_ID_FIELD] = next_id
                    signature_to_id[signature] = next_id
                    merged_data.append(new_record)
                    print(f"    添加新数据 (id={next_id}, config_id={record.get(CONFIG_ID_FIELD, 'N/A')})")
                    next_id += 1
        
        except FileNotFoundError:
            print(f"  错误: 文件 {file_path} 不存在，跳过")
        except json.JSONDecodeError as e:
            print(f"  错误: 文件 {file_path} JSON解析失败: {e}，跳过")
        except Exception as e:
            print(f"  错误: 处理文件 {file_path} 时发生异常: {e}，跳过")
    
    # 保存合并后的数据
    print(f"\n合并完成，共 {len(merged_data)} 条唯一数据")
    print(f"保存到文件: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    
    print("保存成功！")


def main():

    
    merge_json_files(INPUT_FILES, OUTPUT_FILE)


if __name__ == '__main__':
    main()

