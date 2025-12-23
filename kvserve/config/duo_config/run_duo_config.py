import argparse
import pandas as pd
from transformers import AutoModelForCausalLM
from kvserve.engine.logger import LogLevel, set_log_level, log_info
from kvserve.config.duo_config.get_config import DuoConfigGenerator

if __name__ == "__main__":
    # Set log level
    level = LogLevel["INFO"]
    set_log_level(level)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Get DuoAttention scores",
    )
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--q_len", type=int, default=1024, help="Query length")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )

    model_name = args.model_name.split("/")[-1]
    
    # Use the class method to compute scores
    scores = DuoConfigGenerator.duo_attention_on_the_fly(model, args.num_samples, args.q_len, args.max_tokens)
    
    log_info(f"{model_name} DuoAttention scores: \n{scores}")

    # Save scores to CSV file
    output_filename = f"./{model_name}_scores.csv"
    df = pd.DataFrame(scores)
    df.to_csv(output_filename, header=False, index=False)
    log_info(f"Scores successfully saved to {output_filename}")
