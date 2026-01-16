# Complete implementation of step functions for PD separation
# Based on vLLM 0.10.1 API

import torch
import time
from typing import Dict, List, Any
from kvserve.engine.utils import StepOutput
from kvserve.engine.logger import log_debug, log_warning, log_error


def step_prefill_impl(worker, batched_requests, kv_block_tables):
    """
    Execute prefill step: process full sequence and generate KV cache + first token
    Real implementation using model_runner.execute_model (vLLM 0.10.1 API)
    """
    try:
        with torch.inference_mode():
            from vllm.sequence import SequenceData, SequenceGroupMetadata
            from vllm import SamplingParams
            
            # Create SequenceGroupMetadata objects for vLLM 0.10.1
            seq_group_metadata_list = []
            expanded_tokens_map = {}  # Map request_id -> expanded_prompt_token_ids
            
            for idx, request in enumerate(batched_requests.requests):
                # Generate a unique seq_id (use hash of request_id)
                seq_id = hash(request.request_id) % (2**31)
                
                # Get prompt token ids
                prompt_token_ids = list(request.prompt_token_ids) if request.prompt_token_ids else []
                
                # Create SequenceData using from_seqs (vLLM 0.10.1 API)
                seq_data = SequenceData.from_seqs(
                    prompt_token_ids=prompt_token_ids,
                    output_token_ids=None,
                )
                
                # Create sampling params (following vLLM defaults)
                # Handle do_sample: if False, use temperature=0.0 for greedy sampling
                sampling_temp = 0.0 if not request.do_sample else request.temperature
                
                # Build sampling kwargs with vLLM-compatible parameters
                # Note: top_k in vLLM uses 0 or -1 to disable, not distinction needed
                sampling_kwargs = {
                    'temperature': sampling_temp,
                    'top_p': request.top_p,
                    'top_k': request.top_k if request.top_k > 0 else -1,
                    'max_tokens': request.max_tokens if request.max_tokens is not None else 16,  # vLLM default
                }
                
                # Add stop sequences if provided
                if request.stop:
                    sampling_kwargs['stop'] = request.stop
                
                log_debug(f"[Worker] Prefill SamplingParams for {request.request_id}: temp={sampling_temp}, max_tokens={sampling_kwargs['max_tokens']}, stop={request.stop}")
                sampling_params = SamplingParams(**sampling_kwargs)
                
                # Get block table for this request (if available)
                # kv_block_tables is a dict: {request_id: [block_ids...]}
                block_table = kv_block_tables.get(request.request_id, []) if kv_block_tables else []
                
                # Create SequenceGroupMetadata (vLLM 0.10.1)
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=str(request.request_id),
                    is_prompt=True,
                    seq_data={seq_id: seq_data},
                    sampling_params=sampling_params,
                    block_tables={seq_id: block_table},
                    do_sample=True,
                    pooling_params=None,
                    token_chunk_size=len(prompt_token_ids),
                    lora_request=None,
                    computed_block_nums=[],
                    multi_modal_data=None,  # PD separation doesn't handle multimodal
                    multi_modal_placeholders=None,
                )
                
                seq_group_metadata_list.append(seq_group_metadata)
            
            # Prepare model input (vLLM 0.10.1)
            finished_requests_ids = []
            
            model_input = worker.model_runner.prepare_model_input(
                seq_group_metadata_list, 
                virtual_engine=0,
                finished_requests_ids=finished_requests_ids
            )
            
            # Execute model (vLLM 0.10.1)
            # KV cache is already bound to Attention layers via bind_kv_cache()
            # vLLM accesses it directly through Attention.kv_cache[virtual_engine]
            # 
            # For TP>1: Only rank 0 will produce sampling outputs
            # Non-rank-0 workers participate in forward pass but may fail at sampling
            try:
                seq_outs = worker.model_runner.execute_model(model_input, [], None)
            except AssertionError as e:
                # For TP>1, non-rank-0 workers may fail at sampling (logits is None)
                # This is expected behavior - only rank 0 produces outputs
                if worker.tp_rank > 0:
                    log_debug(f"[Worker-{worker.worker_id}] TP rank {worker.tp_rank} skipped sampling (expected)")
                    seq_outs = []  # Return empty for non-rank-0
                else:
                    raise  # Rank 0 should not fail
            
            # Extract generated tokens
            generated_tokens = []
            if seq_outs and len(seq_outs) > 0:
                for output in seq_outs[0]:
                    if hasattr(output, 'samples') and output.samples:
                        generated_tokens.append(output.samples[0].output_token)
                    else:
                        generated_tokens.append(1)  # Fallback
            
            # Return outputs
            outputs = []
            for i, request in enumerate(batched_requests.requests):
                token_id = generated_tokens[i] if i < len(generated_tokens) else 1
                # Check EOS for prefill output
                eos_token_ids = []
                if hasattr(worker.model_runner.model_config, 'hf_config'):
                    hf_config = worker.model_runner.model_config.hf_config
                    if hasattr(hf_config, 'eos_token_id'):
                        if isinstance(hf_config.eos_token_id, list):
                            eos_token_ids = hf_config.eos_token_id
                        else:
                            eos_token_ids = [hf_config.eos_token_id]
                
                finished = eos_token_ids and token_id in eos_token_ids
                
                output = StepOutput(
                    request_id=request.request_id,
                    output_token_ids=[token_id],
                    finished=finished,
                )
                outputs.append(output)
            
            return outputs, expanded_tokens_map
            
    except Exception as e:
        log_error(f"[Worker] Error in step_prefill: {e}")
        import traceback
        traceback.print_exc()
        # Return placeholder outputs on error
        outputs = [StepOutput(request_id=r.request_id, output_token_ids=[1], finished=False) 
                for r in batched_requests.requests]
        return outputs, {}


def step_decode_impl(worker, batched_requests, kv_block_tables):
    """
    Execute decode step: autoregressive generation
    Real implementation using model_runner.execute_model (vLLM 0.10.1 API)
    """
    try:
        with torch.inference_mode():
            from vllm.sequence import SequenceData, SequenceGroupMetadata
            from vllm import SamplingParams
            
            # 🔄 MULTI-STREAM: Sync communication stream before compute
            # This ensures KV transfer is complete before we use the KV cache
            if hasattr(worker, 'comm_stream') and worker.comm_stream is not None:
                sync_result = worker.sync_comm_stream()
                if sync_result.get('elapsed', 0) > 0.001:  # > 1ms
                    log_debug(f"[Decode] Waited {sync_result['elapsed']*1000:.2f}ms for KV transfer")
            
            # Track decode steps
            if not hasattr(worker, '_decode_steps'):
                worker._decode_steps = {}
            
            # Create SequenceGroupMetadata objects for vLLM 0.10.1
            seq_group_metadata_list = []
            for idx, request in enumerate(batched_requests.requests):
                req_id = request.request_id
                worker._decode_steps[req_id] = worker._decode_steps.get(req_id, 0) + 1
                
                # Generate a unique seq_id (use hash of request_id)
                seq_id = hash(request.request_id) % (2**31)
                
                # For decode, we need prompt tokens + all output tokens so far
                prompt_tokens = request.prompt_token_ids if request.prompt_token_ids else []
                output_tokens = request.output_token_ids if request.output_token_ids else []
                
                # Create SequenceData using from_seqs (vLLM 0.10.1 API)
                seq_data = SequenceData.from_seqs(
                    prompt_token_ids=prompt_tokens,
                    output_token_ids=output_tokens,
                )
                
                # CRITICAL: For decode, mark all EXISTING tokens as computed, except the LAST one
                num_prompt = len(prompt_tokens)
                num_output = len(output_tokens)
                total_tokens = num_prompt + num_output
                
                # CRITICAL FIX: If no output tokens, this means prefill failed or KV transfer issue
                if num_output == 0:
                    log_warning(f"[Decode] {req_id} has no output tokens from prefill")
                    log_debug(f"  prompt_tokens={num_prompt}, output_tokens={num_output}")
                    log_debug(f"  request.output_token_ids={request.output_token_ids}")
                    # Don't skip - create metadata anyway, vLLM will handle it
                    # But log the warning
                
                # Mark tokens as computed: all tokens except the last one
                if total_tokens > 0:
                    seq_data.update_num_computed_tokens(total_tokens - 1)
                
                # Create sampling params (following vLLM defaults)
                # Handle do_sample: if False, use temperature=0.0 for greedy sampling
                sampling_temp = 0.0 if not request.do_sample else request.temperature
                
                # Build sampling kwargs with vLLM-compatible parameters
                # Note: top_k in vLLM uses 0 or -1 to disable, not distinction needed
                sampling_kwargs = {
                    'temperature': sampling_temp,
                    'top_p': request.top_p,
                    'top_k': request.top_k if request.top_k > 0 else -1,
                    'max_tokens': request.max_tokens if request.max_tokens is not None else 16,  # vLLM default
                }
                
                # Add stop sequences if provided
                if request.stop:
                    sampling_kwargs['stop'] = request.stop
                
                log_debug(f"[Worker] Decode SamplingParams for {request.request_id}: temp={sampling_temp}, max_tokens={sampling_kwargs['max_tokens']}, stop={request.stop}")
                sampling_params = SamplingParams(**sampling_kwargs)
                
                # Get block table for this request
                block_table = kv_block_tables.get(request.request_id, []) if kv_block_tables else []
                
                # Validate block allocation
                seq_len = len(request.prompt_token_ids) + len(request.output_token_ids)
                blocks_needed = (seq_len + worker.block_size - 1) // worker.block_size
                
                if len(block_table) < blocks_needed:
                    log_error(f"[Decode] {request.request_id} needs {blocks_needed} blocks, but only has {len(block_table)} blocks")
                
                # Create SequenceGroupMetadata (for decode, is_prompt=False)
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=str(request.request_id),
                    is_prompt=False,  # Decode stage
                    seq_data={seq_id: seq_data},
                    sampling_params=sampling_params,
                    block_tables={seq_id: block_table},
                    do_sample=True,
                    pooling_params=None,
                    token_chunk_size=1,  # Decode one token at a time
                    lora_request=None,
                    computed_block_nums=[],
                    multi_modal_data=None,  # PD separation doesn't handle multimodal
                    multi_modal_placeholders=None,
                )
                seq_group_metadata_list.append(seq_group_metadata)
            
            # Check if we have any requests to process
            if not seq_group_metadata_list:
                log_warning(f"[Worker] seq_group_metadata_list is empty, no requests to decode")
                # Return empty outputs but don't fail
                return []
            
            # Prepare model input (vLLM 0.10.1)
            finished_requests_ids = []
            try:
                model_input = worker.model_runner.prepare_model_input(
                    seq_group_metadata_list,
                    virtual_engine=0,
                    finished_requests_ids=finished_requests_ids
                )
            except Exception as e:
                log_error(f"[Worker] Error in prepare_model_input: {e}")
                import traceback
                traceback.print_exc()
                return []
            
            # Execute model (vLLM 0.10.1)
            # KV cache is already bound to Attention layers via bind_kv_cache()
            # For TP>1: Only rank 0 will produce sampling outputs
            step_start_time = time.time()
            try:
                seq_outs = worker.model_runner.execute_model(model_input, [], None)
            except AssertionError as e:
                # For TP>1, non-rank-0 workers may fail at sampling (logits is None)
                if worker.tp_rank > 0:
                    log_debug(f"[Worker-{worker.worker_id}] TP rank {worker.tp_rank} skipped sampling (expected)")
                    return []  # Return empty for non-rank-0
                else:
                    log_error(f"[Worker] Error in execute_model: {e}")
                    raise
            except Exception as e:
                log_error(f"[Worker] Error in execute_model: {e}")
                import traceback
                traceback.print_exc()
                return []
            step_end_time = time.time()
            
            # Extract generated tokens
            generated_tokens = []
            if seq_outs and len(seq_outs) > 0:
                for output in seq_outs[0]:
                    if hasattr(output, 'samples') and output.samples:
                        generated_tokens.append(output.samples[0].output_token)
                    else:
                        generated_tokens.append(1)  # Fallback
            
            log_debug(f"[Worker] Decode generated {len(generated_tokens)} tokens for {len(seq_group_metadata_list)} requests")
            
            # Create outputs - match by request_id from metadata
            outputs = []
            request_map = {req.request_id: req for req in batched_requests.requests}
            
            # Match outputs to requests that were processed
            for i, metadata in enumerate(seq_group_metadata_list):
                req_id = metadata.request_id
                request = request_map.get(req_id)
                if not request:
                    log_warning(f"[Worker] Could not find request {req_id} in batch")
                    continue
                
                token_id = generated_tokens[i] if i < len(generated_tokens) else 1
                output_tokens = (request.output_token_ids or []) + [token_id]
                
                # Check if finished based on max_tokens or EOS
                # Get EOS token IDs from tokenizer (handles different models)
                eos_token_ids = []
                if hasattr(worker.model_runner.model_config, 'hf_config'):
                    hf_config = worker.model_runner.model_config.hf_config
                    if hasattr(hf_config, 'eos_token_id'):
                        if isinstance(hf_config.eos_token_id, list):
                            eos_token_ids = hf_config.eos_token_id
                        else:
                            eos_token_ids = [hf_config.eos_token_id]
                
                if request.max_tokens is not None and len(output_tokens) >= request.max_tokens:
                    finished = True
                elif eos_token_ids and token_id in eos_token_ids:
                    finished = True  # Finish on any EOS token
                else:
                    finished = False
                
                log_debug(f"[Worker] Decode output for {req_id}: new_token={token_id}, total_tokens={len(output_tokens)}, finished={finished}")
                
                output = StepOutput(
                    request_id=req_id,
                    output_token_ids=output_tokens,
                    finished=finished,
                    step_start_time=step_start_time,
                    step_end_time=step_end_time,
                    num_output_tokens=1,
                )
                outputs.append(output)
            
            log_debug(f"[Worker] Returning {len(outputs)} decode outputs")
            return outputs
            
    except Exception as e:
        log_error(f"[Worker] FATAL Error in step_decode: {e}")
        import traceback
        traceback.print_exc()
        # Return placeholder outputs on error
        outputs = [StepOutput(request_id=r.request_id, output_token_ids=[1], finished=False) 
                for r in batched_requests.requests]
        return outputs

