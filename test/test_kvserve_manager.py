import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from kvserve.manager import CompressionManager, CompressionConfig
from kvserve.transformer import KVServeTransformer
from kvserve.quantizer import KVServeQuantizer
from kvserve.codec import KVServeCodec

MODEL_PATH = "/root/workspace/models/Llama-3.1-8B-Instruct"

# load dataset and sample a text
dataset = load_dataset("Xnhyacinth/LongBench", "2wikimqa", split="test").to_pandas()
max_length_idx = dataset['length'].idxmax()
text = dataset.loc[max_length_idx, "context"]

# load tokenizer and encode the text
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
inputs["input_ids"] = inputs["input_ids"][:, :4096]

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    use_cache=True,
    # output_attentions=True,
    attn_implementation="flash_attention_2",
)

# evaluate the model and run the generation
model.eval()
outputs = model.generate(
    **inputs,
    max_new_tokens=1,
    return_dict_in_generate=True,
)

keys = [kv[0] for kv in outputs.past_key_values]
values = [kv[1] for kv in outputs.past_key_values]

# Configure compression with quantizer
config = CompressionConfig(
    enabled=True,
    pipeline=["transformer", "quantizer", "codec"],
    transformer_config={
        "transform_type": "hadamard",
        "seed": 0x3333,
    },
    quantizer_config={
        "model_name": "Llama-3.1-8B-Instruct",
        "hybrid_ratio": 0.5,
        "high_key_max_value": 6,
        "high_value_max_value": 6,
        "low_key_max_value": 4,
        "low_value_max_value": 4,
        "axis_key": "channel",
        "axis_value": "token",
        "split_type": "head",
    },
    codec_config={
        "codec_type": "nvcomp",
        "nvcomp_algorithm": "ANS",
        "data_type": "|u1",
        # "algorithm_type": "0",
    },
    min_compress_size=1024,
)

# Create compression manager
manager = CompressionManager(
    config=config,
    transformer=KVServeTransformer,
    quantizer=KVServeQuantizer,
    codec=KVServeCodec,
)

new_keys = []
new_values = []
original_size = 0
compressed_size = 0
for i in range(len(keys)):
    keys[i] = keys[i].transpose(1, 2).reshape(-1, 32, 8, 128)
    values[i] = values[i].transpose(1, 2).reshape(-1, 32, 8, 128)
    temp_keys = keys[i]
    temp_values = values[i]
    temp_kv_cache = torch.stack([temp_keys, temp_values], dim=0)
    
    # Compress KV cache
    compressed = manager.compress(
        layer_id=i,
        kv_data=temp_kv_cache,  # [2, num_blocks, block_size, num_heads, head_size]
        request_id="req_001",
        metadata={"block_indices": [0, 1, 2]},
    )
    original_size += compressed.original_size
    compressed_size += compressed.compressed_size
    # Decompress KV cache
    decompressed = manager.decompress(
        layer_id=i,
        compressed_data=compressed,
    )
    new_keys.append(decompressed[0])
    new_values.append(decompressed[1])

keys = torch.stack([key.cpu() for key in keys], dim=0)
values = torch.stack([value.cpu() for value in values], dim=0)
new_keys = torch.stack([key.cpu() for key in new_keys], dim=0)
new_values = torch.stack([value.cpu() for value in new_values], dim=0)

# Calculate cosine similarity between original and decompressed KV cache
key_similarity = torch.nn.functional.cosine_similarity(keys.flatten(), new_keys.flatten(), dim=0)
value_similarity = torch.nn.functional.cosine_similarity(values.flatten(), new_values.flatten(), dim=0)
print(f"Key cosine similarity: {key_similarity.item():.2f}")
print(f"Value cosine similarity: {value_similarity.item():.2f}")
print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
print(f"Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
print(f"Compression ratio: {original_size / compressed_size:.2f}")