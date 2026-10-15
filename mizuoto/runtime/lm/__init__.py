"""The LM toolkit: decoder blocks, KV caches and generation loops.

- ``qwen``: the Qwen2/Qwen3-family decoder (RMSNorm, RoPE, GQA, SwiGLU, optional QKV bias and q/k
  norm), with transformers' exact numerics: its SDPA call, mask rules and RoPE computation. A model
  ported onto it gives transformers' results bit for bit on the CPU in fp32.
- ``cache``: ``DynamicCache``, which grows by concatenation as transformers' does.
"""
