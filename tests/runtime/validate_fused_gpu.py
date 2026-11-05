# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the KVServe project
"""Real default-codec GPU gate across model geometries and partial frames."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
os.environ['KVSERVE_COMPRESSION_BACKEND'] = 'cpp'
import torch
from kvserve_v1.compression.runtime import CacheSpec, CompressionConfig, CompressionRuntime, build_memory_plan
from kvserve_v1.compression.runtime.fused_runtime import create_runtime

p = argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
device=torch.device('cuda',0);torch.cuda.set_device(device);torch.manual_seed(117)
profiles=[(2,1,128,64),(3,2,128,256),(28,4,128,1024),(32,8,128,1024),
          (2,16,128,64),(2,4,64,64),(2,4,256,64),(3,4,128,80),(2,3,128,64)]
result={'passed':False,'profiles':[]}
try:
 for layers,heads,dim,capacity in profiles:
  config=CompressionConfig.from_extra_config({'compression':{'enable-compression-pipeline':True,'slot_count':2,'logical_blocks_per_chunk':capacity}})
  specs={f'layer.{i}':CacheSpec(i,1,heads,dim,torch.bfloat16) for i in range(layers)}
  caches={name:tuple(torch.full((capacity*2+64,heads,1,dim),17,dtype=torch.bfloat16,device=device) for _ in (0,1)) for name in specs}
  for pair in caches.values():
   for tensor in pair:tensor[:capacity].normal_()
  plan=build_memory_plan(config,specs,blocks_per_logical=1,num_layers=layers,device=device)
  ref=CompressionRuntime(plan,'HND',1,device)
  rt=create_runtime(plan,'HND',1,device,output_slots=2,graph_lengths=(16,capacity))
  row={'layers':layers,'heads':heads,'dim':dim,'capacity':capacity,'backend':type(rt).__name__,'cases':[]}
  try:
   ref.bind_kv_caches(caches,{name:0 for name in specs});rt.bind_kv_caches(caches,{name:0 for name in specs})
   for n in sorted(set([1,17,min(332,capacity),capacity])):
    if n>capacity:continue
    ids=torch.randperm(capacity)[:n].tolist();dst=list(range(capacity+32,capacity+32+n))
    for part in (0,1):
     original=ref.encode(0,[ids],part,valid_tokens_by_group={0:n});original.event.synchronize()
     qref=ref.slots[0].arena_b[ref.layout.u8_base:ref.layout.u8_base+original.u8_bytes].clone()
     auxref=ref.slots[0].aux[:ref.aux_bytes(part,{0:n})].clone()
     ref.decode(0,[dst],part,original.codec_bytes,original.u8_bytes,valid_tokens_by_group={0:n}).synchronize()
     expected=torch.stack([caches[name][part][dst] for name in specs])
     for pair in caches.values():pair[part][capacity:].fill_(17)
     payload,auxiliary=rt.output_slots[0]
     encoded=rt.encode_into(0,[ids],part,valid_tokens_by_group={0:n},output=payload,auxiliary=auxiliary);encoded.event.synchronize()
     size=encoded.codec_bytes;assert size>0 and encoded.raw_tail_bytes==0
     q=rt.slots[0].arena_b[rt.layout.u8_base:rt.layout.u8_base+encoded.u8_bytes]
     width=heads*dim
     if row['backend']=='FusedCompressionRuntime':
      bucket=(n+15)//16*16
      assert torch.equal(q.view(layers,bucket,width)[:,:n],qref.view(layers,capacity,width)[:,:n]),(row,n,part,'quant')
     else:assert torch.equal(q,qref),(row,n,part,'quant-standard')
     assert torch.equal(auxiliary[:auxref.numel()],auxref),(row,n,part,'aux')
     rt.slots[0].arena_a[:size].copy_(payload[:size]);rt.slots[0].aux[:auxref.numel()].copy_(auxiliary[:auxref.numel()]);torch.cuda.synchronize(device)
     rt.decode(0,[dst],part,size,encoded.u8_bytes,valid_tokens_by_group={0:n}).synchronize()
     actual=torch.stack([caches[name][part][dst] for name in specs])
     assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)),(row,n,part,'restored')
     for pair in caches.values():pair[part][capacity:].fill_(17)
     row['cases'].append({'tokens':n,'part':part,'quant_aux_restored_exact':True,'codec_bytes':size})
   print({k:v for k,v in row.items() if k!='cases'},flush=True)
   result['profiles'].append(row)
  finally:rt.shutdown();ref.shutdown()
 result['passed']=True
finally:a.output.write_text(json.dumps(result,indent=2)+'\n')
