import sys
import os
import torch
import pandas as pd
from typing import Any
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import get_dtype, stop_sequences_criteria
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from offline_search.src.cache.cache_utils import CustomCacheConfig, CustomCache

from offline_search.src.models.modeling_llama import enable_custom_llama_attention
from offline_search.src.models.modeling_qwen import enable_custom_qwen2_attention

# 使用 @register_model 装饰器将模型注册到 lm_eval
@register_model("my_custom_model")
class CustomLMEval(HFLM):

    # 重写 _create_model 方法
    def _create_model(
        self,
        pretrained: str,
        revision: str | None = "main",
        dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool | None = True,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: bool | None = False,
        gpus: int | None = None,
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        # PEFT, delta weights and quantization options
        peft: str | None = None,
        delta: str | None = None,
        autogptq: bool | str | None = False,
        gptqmodel: bool | None = False,
        gguf_file: str | None = None,
        subfolder: str = "",
        **kwargs,
    ) -> None:
        """
        这个方法在 HFLM 的 __init__ 中被调用，
        用来加载实际的 Hugging Face Transformers 模型。
        """
        
        print(f"\nLoading custom model using [CustomLMEval] from: {pretrained}")
        
        model_kwargs = kwargs or {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map"),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        # load model's config
        model_config = AutoConfig.from_pretrained(pretrained)

        self.cache_type = kwargs.pop("cache_type", "default")
        self.comp_cr = kwargs.get("comp_cr", False)
        self.cr_list = kwargs.pop("cr_list", None)

        match self.cache_type:
            case "custom":
                # Custom KV Cache from offline_search
                scores = kwargs.pop("scores", None)
                assert scores is not None, "Models scores must be provided, check your config."

                self.cache_config = CustomCacheConfig(
                    transform_type=kwargs.pop("transform_type", "none"),
                    scores=scores,
                    heads_selection=kwargs.pop("heads_selection", 0.5),
                    high_key_max_value=kwargs.pop("high_key_max_value", 64),
                    high_value_max_value=kwargs.pop("high_value_max_value", 64),
                    low_key_max_value=kwargs.pop("low_key_max_value", 32),
                    low_value_max_value=kwargs.pop("low_value_max_value", 32),
                    axis_key=kwargs.pop("axis_key", [2]),
                    axis_value=kwargs.pop("axis_value", [1, 3]),
                    comp_cr=kwargs.pop("comp_cr", False),
                )

                print("\n============== Using custom cache config. ==============\n")

            case "kivi":
                # KIVI KV Cache from huggingface and KIVI's paper
                from offline_search.src.cache.kivi_utils import KIVICacheConfig

                self.cache_config = KIVICacheConfig(
                    nbits=kwargs.pop("nbits", 4),
                    axis_key=kwargs.pop("axis_key", 1),
                    axis_value=kwargs.pop("axis_value", 0),
                    q_group_size=kwargs.pop("q_group_size", 64),
                    residual_length=kwargs.pop("residual_length", 128),
                    comp_cr=kwargs.pop("comp_cr", False),
                )
                
                print("\n============== Using kivi cache config. ==============\n")

            case "cachegen":
                # CacheGen KV Cache from it's paper
                from offline_search.src.cache.cachegen_utils import CacheGenCacheConfig
                num_layers = getattr(model_config, "num_hidden_layers", getattr(model_config, "n_layer", None))
                assert num_layers is not None, "Model layers must be provided, check your model's config.json file or set it manually."
                self.cache_config = CacheGenCacheConfig(
                    model_layers=num_layers,
                    quantization_level=kwargs.pop("quantization_level", 1),
                    high_max_value=kwargs.pop("high_max_value", 32),
                    mid_max_value=kwargs.pop("mid_max_value", 16),
                    low_max_value=kwargs.pop("low_max_value", 12),
                    comp_cr=kwargs.pop("comp_cr", False),
                )
                print("\n============== Using cachegen cache config. ==============\n")

            case "duoattn":
                from offline_search.src.cache.duoattn_utils import DuoAttentionCacheConfig

                scores = kwargs.pop("scores", None)
                assert scores is not None, "Models scores must be provided, check your config."

                self.cache_config = DuoAttentionCacheConfig(
                    scores=scores,
                    heads_selection=kwargs.pop("heads_selection", 0.5),
                    sink_size=kwargs.pop("sink_size", 128),
                    recent_size=kwargs.pop("recent_size", 256),
                    comp_cr=kwargs.pop("comp_cr", False),
                )
                print("\n============== Using duoattn cache config. ==============\n")
                
            case "default":
                self.cache_config = None
                kwargs.pop("comp_cr", False)
                print("No scores provided, using default cache config.")
            case _:
                raise ValueError(f"Invalid cache type: {kwargs.get('cache_type')}")


        enable_custom_llama_attention()
        enable_custom_qwen2_attention()
        
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype = get_dtype(kwargs.get("dtype", "auto")),
            **kwargs
        )
        # self._model.generation_config.temperature = None
        # self._model.generation_config.top_p = None
        # self._model.generation_config.top_k = None        

    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )

        match self.cache_type:
            case "custom":
                past_key_values = CustomCache(cache_config=self.cache_config)
                # print("\n============== Using custom past key values. ==============\n")
            case "kivi":
                from offline_search.src.cache.kivi_utils import KIVICache

                past_key_values = KIVICache(cache_config=self.cache_config)
                # print("\n============== Using kivi past key values. ==============\n")
            case "cachegen":
                from offline_search.src.cache.cachegen_utils import CacheGenCache

                past_key_values = CacheGenCache(cache_config=self.cache_config)
                # print("\n============== Using cachegen past key values. ==============\n")
            case "duoattn":
                from offline_search.src.cache.duoattn_utils import DuoAttentionCache

                past_key_values = DuoAttentionCache(cache_config=self.cache_config)
                # print("\n============== Using duoattn past key values. ==============\n")
            case "default":
                past_key_values = None
                # print("\n============== Using default past key values. ==============\n")
            case _:
                raise ValueError(f"Invalid cache type: {self.cache_type}")

        # ------ Generate ------
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            outputs = self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                past_key_values=past_key_values, # for custom cache
                **generation_kwargs,
            )    
            if self.comp_cr and self.cr_list is not None:
                self.cr_list.append(past_key_values.compression_ratio)
            return outputs
        
