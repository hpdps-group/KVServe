import os
from vllm import LLM, SamplingParams

# 配置 GPU 绑定（这里用 GPU 0,1 做 TP=2）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_USE_CUSTOM_ALLREDUCE"] = "0"  # 关闭自定义 allreduce，减少 P2P 检查

MODEL_PATH = "/root/ssd/Llama3.1-8B-Instruct"

def main():
    # 创建 LLM（TP=2）
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.75,
        max_model_len=2048,
        dtype="float16",
        enforce_eager=True,
        disable_log_stats=True,
        disable_custom_all_reduce=True,
        max_num_seqs=4,  # 小批次即可
    )

    prompts = ["Hello, how are you?"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

    outputs = llm.generate(prompts, sampling_params)
    for out in outputs:
        print("=== OUTPUT ===")
        print(out.outputs[0].text)

if __name__ == "__main__":
    main()