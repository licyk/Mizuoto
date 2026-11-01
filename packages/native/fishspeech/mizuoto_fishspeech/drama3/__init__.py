"""Fish Audio Drama 3 (fish-speech 0f8a12e8, ``fish_speech/models/text2semantic/fish_omni``; Fish Audio Community
License 1.0), ported natively.

A Qwen3.5-MoE backbone (``backbone``: forty layers, three of every four Gated DeltaNet linear attention and every fourth
gated softmax attention, each followed by 256 routed experts of which 8 run per token, plus a gated shared expert)
reads the prompt (``prompt``: a chat whose voice turns are frames of ten codes) and predicts each frame's main token; a
codebook tower (``tower``: 4 shared + 9 residual transformer layers attending over time) then draws the frame's other
nine codes. S2's codec turns the frames into 44.1 kHz audio. ``generate`` runs it chunk by chunk as upstream's
``generate_long_drama3``; ``engine`` is the loaded engine.
"""
