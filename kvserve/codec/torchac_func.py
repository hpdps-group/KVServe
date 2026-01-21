"""
TorchAC entropy codec for KV cache compression

Implements arithmetic coding using CUDA kernels (torchac_cuda) to compress and
decompress KV tensors. Mirrors nvcomp_func.py style with clear encode/decode
interfaces for KVServe.
"""

import torch
from kvserve.engine.logger import log_info, log_error, log_warning, log_debug
try:
    import torchac_cuda
except ImportError:
    log_error("[WARNING] torchac_cuda not found, using CPU-only implementation, please refer to https://github.com/UChi-JCL/CacheGen for installation")

from kvserve.manager import EasyDist

class TorchACCodec:
    """
    TorchAC codec wrapper

    Provides encode/decode helpers around CUDA-accelerated arithmetic coding
    for KV cache tensors.

    This torchac_cuda is referred from https://github.com/UChi-JCL/CacheGen.
    """

    def __init__(
        self, 
        **kwargs
    ) -> None:

        pass

    def encode(
        self, 
        tensor: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Encode (compress) KV cache tensor with TorchAC arithmetic coding

        Permutes tensor into expected layout, builds CDFs, encodes tokens in
        chunks, and packs bytestream plus metadata via EasyDist.

        Args:
            tensor: KV cache tensor to compress, shape expected by KVServe
            **kwargs: Reserved for future options

        Returns:
            Compressed tensor (EasyDist-packed) containing bytestream chunks and CDFs
        """
        tensor = tensor.permute(0, 1, 3, 4, 2, 5)
        num_layers, kv, num_blocks, block_size, num_heads, head_size = tensor.shape
        keys = tensor[:, 0].reshape(num_layers, num_blocks * block_size, num_heads * head_size).view(torch.int8)
        values = tensor[:, 1].reshape(num_layers, num_blocks * block_size, num_heads * head_size).view(torch.int8)        
        encode_input = torch.cat((keys, values), dim=0)

        nlayers, ntokens, nchannels = encode_input.shape

        # Compute the maximum value needed for CDF calculation
        max_raw_value = max(keys.max().item(), values.max().item())
        max_value = 1
        while max_value < max_raw_value:
            max_value *= 2
        new_cdf_key = torchac_cuda.calculate_cdf(keys, max_value)
        new_cdf_value = torchac_cuda.calculate_cdf(values, max_value)
        cdf_int = torch.cat([new_cdf_key, new_cdf_value])

        # Set the buffer size per channel
        buffer_size_per_channel = 256
        output_buffer = torch.zeros(
                (nlayers, nchannels, buffer_size_per_channel), 
                dtype=torch.uint8, 
                device=encode_input.device)
        output_lengths = torch.zeros(
                (nlayers, nchannels), 
                dtype=torch.int32, 
                device=encode_input.device)

        data_chunks = []
        for i in range(0, ntokens, 256):
            start = i
            end = min(i + 256, ntokens)
            bytestream = self.encode_ntokens(
                cdf_int,
                encode_input[:, start:end, :],
                output_buffer,
                output_lengths
            )
            data_chunks.append({
                "bytestream": bytestream, 
                "bytestream_lengths": output_lengths.clone(),
                "ntokens": end - start
            })

        compressed_data = {
                "data_chunks": data_chunks,
                "cdf_int": cdf_int,
        }
        compressed_tensor, _ = EasyDist.pack_object(compressed_data)
        return compressed_tensor
        
    def decode(
        self, 
        compressed_tensor: torch.Tensor,
        original_dtype: str,
        original_shape: list,
        device: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode (decompress) KV cache tensor from TorchAC bytestream

        Unpacks metadata, reconstructs tokens via CUDA decode, reshapes, and
        permutes back to original layout and dtype.

        Args:
            compressed_tensor: Output from encode()
            original_dtype: Original tensor dtype name (e.g., "bfloat16")
            original_shape: Original tensor shape list
            device: Target device for output tensor
            **kwargs: Reserved for future options

        Returns:
            Tensor restored to original dtype and shape
        """
        compressed_data = EasyDist.unpack_object(compressed_tensor)

        data_chunks = compressed_data["data_chunks"]
        cdf = compressed_data["cdf_int"]

        nlayers, nchannels, _ = cdf.shape
        ntokens = original_shape[3] * original_shape[4]
        output = torch.zeros((nlayers, ntokens, nchannels), dtype=torch.uint8, device=device)

        start = 0
        for data_chunk in data_chunks:
            end = start + data_chunk["ntokens"]
            self.decode_chunk(cdf, data_chunk, output[:, start:end, :])
            start = end

        target_dtype = getattr(torch, original_dtype)
        target_shape = (original_shape[1], original_shape[0], original_shape[3], original_shape[4], original_shape[2], original_shape[5])
        out = output.view(target_dtype).reshape(target_shape)
        return out.permute(1, 0, 4, 2, 3, 5)

    def collect_bytes(self, output_buffer, output_lengths) -> torch.Tensor:
        """
        Collect a byte tensor from the output_buffer + output_lengths
        """
        output_buffer_size = output_buffer.shape[-1]
        flattened_lengths = output_lengths.flatten()
        flattened_buffer = output_buffer.flatten()
        summed_length = (output_buffer_size - flattened_lengths).cumsum(0)
        summed_length = summed_length.roll(1)
        summed_length[0] = 0
        indexes = summed_length.repeat_interleave(flattened_lengths)
        indexes = indexes + torch.arange(len(indexes), device=indexes.device)
        return flattened_buffer[indexes]

    def encode_ntokens(self, cdf_int, encode_input, output_buffer, output_lengths) -> torch.Tensor:
        """
        Input:
            cdf_int: int16 tensor on GPU with shape [nlayers, nchannels, Lp]
            encode_input: int8 tensor on GPU with shape [nlayers, ntokens, nchannels]
            output_buffer: uint8 tensor on GPU with shape [nlayers, nchannels, BUFFER_SIZE]
            output_lengths: int32 tensor on GPU with shape [nlayers, nchannels]
        Returns:
            byte_tensor: the byte tensor
        """
        try:
            torchac_cuda.encode_fast_new(
                    cdf_int,
                    encode_input,
                    output_buffer,
                    output_lengths,
            )
            
        except Exception as e:
            print(f"[ERROR] CUDA kernel failed: {e}")
            print(f"  cdf_int shape: {cdf_int.shape}, dtype: {cdf_int.dtype}")
            print(f"  encode_input shape: {encode_input.shape}, dtype: {encode_input.dtype}")
            print(f"  output_buffer shape: {output_buffer.shape}, dtype: {output_buffer.dtype}")
            raise
        
        byte_tensor = self.collect_bytes(output_buffer, output_lengths)
        return byte_tensor

    def decode_chunk(
            self,
            cdf: torch.Tensor,
            data_chunk: dict,
            target_buffer: torch.Tensor
        ) -> torch.Tensor:
        """
        Write the decode output in target_buffer
        Expected shape: [nlayers (kv in total), ntokens, nchannels]
        """
        #recombined_output = recombine_bytes(bytes_to_tensor(data_chunk.bytestream), data_chunk.bytestream_lengths)
        #torchac_cuda.decode_fast_new(
        #        cdf,
        #        recombined_output,
        #        data_chunk.bytestream_lengths,
        #        target_buffer)
        #bytes_tensor = bytes_to_tensor(data_chunk.bytestream)
        bytes_tensor = data_chunk["bytestream"]
        length_prefsum = data_chunk["bytestream_lengths"].flatten().cumsum(0).reshape(data_chunk["bytestream_lengths"].shape)
        torchac_cuda.decode_fast_prefsum(
                cdf,
                bytes_tensor,
                length_prefsum,
                target_buffer)