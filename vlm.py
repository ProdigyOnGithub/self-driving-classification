"""Shared model code: frozen SigLIP encoder -> trained projector -> Qwen SLM that answers MOVE / STOP.

Used by train.py (GPU) and infer.py (laptop / Orin / RPi).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# The prompt is split around the image. Everything before the image is identical for every frame,
# so at inference its KV cache is computed once and reused (see Decider.prefix_cache).
PROMPT_PRE = ("You are the front camera of a small autonomous vehicle driving on a campus road. "
              "Decide whether it can keep driving forward safely.\nCamera image:")
PROMPT_POST = "\nIs the path ahead clear of obstacles? Answer MOVE or STOP.\nAnswer:"
LABELS = ("MOVE", "STOP")  # index 0 = MOVE, 1 = STOP

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def pick_device(name=None):
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------- vision side
@torch.no_grad()
def encode_images(encoder, processor, images, layer=-2, pool=2):
    """PIL images -> visual tokens [B, T, D].

    layer: which encoder layer to read (-1 = last, with SigLIP's final norm; -2 = second to last).
    pool:  average-pool the token grid by this factor (24x24 -> 12x12 for 384px input).
    """
    p = next(encoder.parameters())
    px = processor(images=images, return_tensors="pt")["pixel_values"].to(p.device, p.dtype)
    out = encoder(pixel_values=px, output_hidden_states=(layer != -1))
    h = out.last_hidden_state if layer == -1 else out.hidden_states[layer]
    if pool > 1:
        b, t, d = h.shape
        s = int(math.isqrt(t))
        h = h.transpose(1, 2).reshape(b, d, s, s)
        h = F.avg_pool2d(h, pool)
        h = h.flatten(2).transpose(1, 2)
    return h


class Projector(nn.Module):
    def __init__(self, vis_dim, llm_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(vis_dim), nn.Linear(vis_dim, llm_dim), nn.GELU(),
                                 nn.Linear(llm_dim, llm_dim))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- LoRA (no peft dependency)
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_A.T @ self.lora_B.T) * self.scale


def add_lora(model, r=16, alpha=32, dropout=0.05, targets=LORA_TARGETS):
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(mod, name, LoRALinear(child, r, alpha, dropout))
    return model


@torch.no_grad()
def merge_lora(model):
    """Fold LoRA into the base weights (for fast inference / export)."""
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                w = child.base.weight
                w += (child.lora_B @ child.lora_A).to(w.dtype) * child.scale
                setattr(mod, name, child.base)
    return model


# ---------------------------------------------------------------- the decider
class Decider(nn.Module):
    """[prompt-pre | projected image tokens | prompt-post] -> Qwen -> scores for ' MOVE' vs ' STOP'."""

    def __init__(self, slm, tokenizer, projector):
        super().__init__()
        self.slm, self.projector = slm, projector
        dev = next(slm.parameters()).device
        self.register_buffer("pre_ids", tokenizer(PROMPT_PRE, return_tensors="pt").input_ids.to(dev), persistent=False)
        self.register_buffer("post_ids", tokenizer(PROMPT_POST, return_tensors="pt").input_ids.to(dev), persistent=False)
        ids = [tokenizer(" " + w).input_ids for w in LABELS]
        assert all(len(i) == 1 for i in ids) and ids[0] != ids[1], f"answer words must be single distinct tokens: {ids}"
        self.answer_ids = [i[0] for i in ids]

    def _readout(self, h):
        w = self.slm.lm_head.weight[self.answer_ids]            # [2, D]
        return h.float() @ w.float().T                           # [B, 2]  (MOVE, STOP)

    def forward(self, feats):
        emb = self.slm.get_input_embeddings()
        vis = self.projector(feats)
        b = vis.shape[0]
        pre = emb(self.pre_ids).expand(b, -1, -1)
        post = emb(self.post_ids).expand(b, -1, -1)
        x = torch.cat([pre, vis.to(pre.dtype), post], dim=1)
        h = self.slm.model(inputs_embeds=x, use_cache=False).last_hidden_state[:, -1]
        return self._readout(h)

    # ---- fast single-frame inference with the prompt prefix cached once
    @torch.no_grad()
    def prefix_cache(self):
        emb = self.slm.get_input_embeddings()
        out = self.slm.model(inputs_embeds=emb(self.pre_ids), use_cache=True)
        return out.past_key_values, self.pre_ids.shape[1]

    @torch.no_grad()
    def forward_cached(self, feats, cache, prefix_len):
        emb = self.slm.get_input_embeddings()
        vis = self.projector(feats)
        post = emb(self.post_ids)
        x = torch.cat([vis.to(post.dtype), post], dim=1)
        out = self.slm.model(inputs_embeds=x, past_key_values=cache, use_cache=True)
        cache.crop(-(cache.get_seq_length() - prefix_len))       # drop this frame, keep the prompt
        return self._readout(out.last_hidden_state[:, -1])


def stop_score(logits):
    """> 0 means the model prefers STOP. Compare against the saved threshold."""
    return logits[:, 1] - logits[:, 0]
