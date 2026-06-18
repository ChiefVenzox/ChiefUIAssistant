"""
backend/app/model/gpt.py
========================
Sıfırdan yazılmış decoder-only Transformer (GPT tarzı). Hazır model/ağırlık yok.
Bileşenler: RMSNorm · RoPE · SwiGLU · (G)MHA + SDPA · KV-cache · grad checkpoint.
Düşük VRAM (6 GB) gözetilerek tasarlandı. (yerelLLM çekirdeğinden doğrulanmış.)
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from app.model.config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(head_dim, max_seq, theta):
    assert head_dim % 2 == 0, "head_dim çift olmalı (RoPE)"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        kv_dim = self.n_kv_head * self.head_dim
        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, kv_dim, bias=False)
        self.wv = nn.Linear(cfg.n_embd, kv_dim, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, attn_mask, past_kv=None, use_cache=False):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if past_kv is not None and past_kv[0] is not None:
            pk, pv = past_kv
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
        new_past = (k, v) if use_cache else None
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        dp = self.dropout if self.training else 0.0
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dp)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.wo(y))
        return y, new_past


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg._ffn_hidden()
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.rms_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask, past_kv=None, use_cache=False):
        h, new_past = self.attn(self.attn_norm(x), cos, sin, attn_mask, past_kv, use_cache)
        x = x + h
        x = x + self.mlp(self.ffn_norm(x))
        return x, new_past


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_layer = cfg.n_layer
        self.block_size = cfg.block_size
        head_dim = cfg.n_embd // cfg.n_head

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_embeddings:
            n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None, past_kvs=None):
        B, T = idx.shape
        use_cache = past_kvs is not None
        past_len = 0
        if use_cache and past_kvs[0] is not None:
            past_len = past_kvs[0][0].size(2)
        assert past_len + T <= self.block_size, (
            f"bağlam taştı: {past_len + T} > block_size={self.block_size}"
        )
        cos = self.rope_cos[past_len:past_len + T]
        sin = self.rope_sin[past_len:past_len + T]
        x = self.drop(self.tok_emb(idx))

        if past_len > 0:
            q_pos = torch.arange(past_len, past_len + T, device=idx.device)
            k_pos = torch.arange(0, past_len + T, device=idx.device)
            attn_mask = (q_pos[:, None] >= k_pos[None, :])
        else:
            attn_mask = None

        new_past = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pkv = past_kvs[i] if use_cache else None
            if self.gradient_checkpointing and self.training and not use_cache:
                x, npkv = checkpoint(block, x, cos, sin, attn_mask, pkv, use_cache,
                                     use_reentrant=False)
            else:
                x, npkv = block(x, cos, sin, attn_mask, pkv, use_cache)
            if use_cache:
                new_past.append(npkv)

        x = self.norm_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-1)
            return logits, loss, None
        logits = self.lm_head(x[:, [-1], :])
        return logits, None, new_past

    def configure_optimizers(self, cfg: GPTConfig, device_type: str):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if cfg.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                print("[optim] bitsandbytes AdamW8bit")
                return bnb.optim.AdamW8bit(groups, lr=cfg.learning_rate,
                                           betas=(cfg.beta1, cfg.beta2))
            except Exception as e:
                print(f"[optim] 8-bit yüklenemedi ({e}); torch AdamW")
        return torch.optim.AdamW(groups, lr=cfg.learning_rate,
                                 betas=(cfg.beta1, cfg.beta2),
                                 fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None,
                 top_p=None, eos_token_id=None, use_cache=True):
        for tok in self.generate_stream(idx, max_new_tokens, temperature, top_k,
                                         top_p, eos_token_id, use_cache):
            idx = torch.cat((idx, tok), dim=1)
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=0.8, top_k=None,
                        top_p=None, eos_token_id=None, use_cache=True):
        self.eval()
        if idx.size(1) > self.block_size:
            idx = idx[:, -self.block_size:]
        past_kvs = [None] * self.n_layer if use_cache else None
        cur_len = idx.size(1)
        next_id = idx
        for step in range(max_new_tokens):
            idx_cond = (idx if step == 0 else next_id) if use_cache else idx[:, -self.block_size:]
            logits, _, past_kvs = self(idx_cond, past_kvs=past_kvs)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = self._filter_logits(logits / temperature, top_k, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            if not use_cache:
                idx = torch.cat((idx, next_id), dim=1)
            cur_len += 1
            yield next_id
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
            if cur_len >= self.block_size:
                break

    @staticmethod
    def _filter_logits(logits, top_k, top_p):
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and 0 < top_p < 1.0:
            s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            s_logits = s_logits.masked_fill(remove, float("-inf"))
            logits = torch.empty_like(logits).scatter_(-1, s_idx, s_logits)
        return logits
