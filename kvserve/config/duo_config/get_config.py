import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from kvserve.engine.logger import log_info

class DuoConfigGenerator:
    """
    Class to generate and retrieve DuoAttention configuration scores.
    """
    
    @staticmethod
    def duo_attention_on_the_fly(model, num_samples=None, q_len=1024, max_tokens=2048):
        """
        Method to quickly compute DuoAttention scores:
        - Compute the mean query and key on num_samples random samples from BookSum. If num_samples is None or <= 0, all samples are used.
        - The input texts are truncated to max_tokens.
        - Repeat the mean query and key q_len times and apply RoPE to get (Q, K)
        - Compute the attention weights for (Q[-1], K) and compute the "area under the cumulated attention curve"
        This method referenced from https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/duo_attention_press.py
        """

        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        num_heads = model.config.num_attention_heads
        num_key_value_heads = model.config.num_key_value_heads
        num_key_value_groups = num_heads // num_key_value_heads

        # Load data
        dataset = load_dataset("kmfoda/booksum", split="train").to_pandas()
        
        if num_samples and num_samples > 0:
            texts = dataset.sample(num_samples, random_state=42)["chapter"].tolist()
            num_texts = num_samples
        else:
            texts = dataset["chapter"].tolist()
            num_texts = len(texts)

        # Initialize variables
        position_ids = torch.arange(q_len).unsqueeze(0)
        scores = torch.zeros((model.config.num_hidden_layers, num_key_value_heads), dtype=torch.float32)

        # Compute scores
        for text in tqdm(texts, desc="Computing DuoAttention scores"):
            with torch.no_grad():
                # Compute hidden states
                inputs = tokenizer(text, return_tensors="pt", max_length=max_tokens, truncation=True).to(model.device)
                hidden_states = list(model(**inputs, output_hidden_states=True).hidden_states[:-1])

                for layer_idx, h in enumerate(hidden_states):
                    module = model.model.layers[layer_idx]
                    d = module.self_attn.head_dim
                    h = module.input_layernorm(h)

                    # Mean query
                    q = module.self_attn.q_proj(h)
                    q = q.view(1, q.shape[1], -1, d)
                    if isinstance(module, (Gemma3Attention, Qwen3Attention)):
                        q = module.q_norm(q)
                    q = q.mean(dim=1, keepdim=True)
                    q = q.repeat(1, q_len, 1, 1).transpose(1, 2)

                    # Mean key
                    k = module.self_attn.k_proj(h)
                    k = k.view(1, k.shape[1], -1, d)
                    if isinstance(module, (Gemma3Attention, Qwen3Attention)):
                        k = module.k_norm(k)
                    k = k.mean(dim=1, keepdim=True)
                    k = k.repeat(1, q_len, 1, 1).transpose(1, 2)

                    # Apply RoPE
                    cos, sin = model.model.rotary_emb(h, position_ids.to(h.device))
                    q, k = apply_rotary_pos_emb(q, k, cos.to(q.device), sin.to(q.device))
                    k = k.repeat_interleave(num_key_value_groups, dim=1)

                    # Compute attention weights for the last token
                    attn_weights = torch.matmul(q[:, :, -1:, :], k.transpose(2, 3)) / (d**0.5)
                    attn_weights = attn_weights.softmax(dim=-1, dtype=torch.float32).squeeze()

                    # Compute score: area under the cumulated attention curve
                    s = torch.cumsum(attn_weights, dim=1, dtype=torch.float32).mean(1)
                    s = s.view(-1, num_key_value_groups).mean(1)

                    # Store the scores
                    scores[layer_idx] += s.cpu() / num_texts

                del hidden_states
                torch.cuda.empty_cache()       

        return scores.numpy()

    @classmethod
    def get_scores_from_csv(cls, model_name: str):
        """
        Retrieve scores from a CSV file in the current directory or the directory of this file.
        Format: <model_basename>_scores.csv
        """
        # Extract the model name from the path
        model_basename = model_name.strip("/").split("/")[-1]
        csv_filename = f"{model_basename}_scores.csv"
        
        # List of paths to check
        paths_to_check = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename), # Directory of this script
            os.path.join(os.getcwd(), csv_filename), # Current working directory
            csv_filename # Relative path
        ]
        
        csv_path = None
        for path in paths_to_check:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path:
            log_info(f"Loading DuoAttention scores from {csv_path}")
            try:
                # Read CSV, assuming no header as per the standard output of run_duo_config
                df = pd.read_csv(csv_path, header=None).dropna()
                return torch.tensor(df.values, dtype=torch.float32)
            except Exception as e:
                log_info(f"Error reading scores file {csv_path}: {e}")
                return None
        else:
            log_info(f"Scores file not found for {model_basename}. Checked: {paths_to_check}")
            return None

