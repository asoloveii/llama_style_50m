from dataclasses import dataclass
import torch, torch.nn as nn
import torch.nn.functional as F


def precompute_rope_embeddings(head_dim: int, max_seq_len: int, theta: float):
    # calculate theta_i = 1 / (base ^ (2i/d))
    inv_freq = 1 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # generate position indices m
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # outer product: m * theta
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # repeat for cos and sin
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    return emb


def apply_rope_to_tensor(x, rope_emb_sliced):
    emb = rope_emb_sliced.unsqueeze(0).unsqueeze(1)
    cos, sin = emb.cos(), emb.sin()

    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_partner = torch.stack([-x_reshaped[..., 1], x_reshaped[..., 0]], dim=-1)
    x_partner = x_partner.flatten(-2)

    return (x * cos + x_partner * sin).type_as(x)


def apply_rope(x, rope_embs):
    # slice embeddings to sequence length
    seq_len = x.shape[1]
    rope_emb_sliced = rope_embs[:seq_len, :]

    # reshape for broadcasting
    emb = rope_emb_sliced.unsqueeze(0).unsqueeze(2)
    cos, sin = emb.cos(), emb.sin()

    # create parter (-y, x)
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_partner = torch.stack([-x_reshaped[..., 1], x_reshaped[..., 0]], dim=-1)
    x_partner = x_partner.flatten(-2)

    return (x * cos + x_partner * sin).type_as(x)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32_000
    max_batch_size: int = 16
    d_model: int = 512
    hidden_dim: int = 1408
    n_layers: int = 12
    n_heads: int = 8
    n_groups: int = 4
    max_seq_len: int = 1024
    theta: float = 10000.0
    eos_token_id: int = 2


class SwiGLU(nn.Module):

    def __init__(self, d_model, hidden_dim):
        super().__init__()

        self.value_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        return self.out_proj(self.value_proj(x) * F.silu(self.gate_proj(x)))


class GQA(nn.Module):

    def __init__(self, args: ModelConfig): 
        super().__init__()

        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.head_dim = args.d_model // args.n_heads
        self.n_groups = args.n_groups
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.kv_proj = nn.Linear(self.d_model, 2*(self.head_dim*self.n_groups), bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.register_buffer(
            "k_cache",
            torch.zeros(self.max_batch_size, self.n_groups, self.max_seq_len, self.head_dim),
            persistent=False
        )

        self.register_buffer(
            "v_cache",
            torch.zeros(self.max_batch_size, self.n_groups, self.max_seq_len, self.head_dim),
            persistent=False
        )

    def forward(self, 
                x: torch.Tensor, 
                rope_embs: torch.Tensor, 
                start_pos: int = 0, 
                use_cache: bool = False):
        bsz, seq, dim = x.shape

        q = self.q_proj(x).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(bsz, seq, 2, self.head_dim, self.n_groups).permute(2, 0, 4, 1, 3)
        k, v = kv[0], kv[1]
        
        current_rope = rope_embs[start_pos:start_pos+seq]
        q = apply_rope_to_tensor(q, current_rope)
        k = apply_rope_to_tensor(k, current_rope)
        
        if use_cache:
            self.k_cache[:bsz, :, start_pos:start_pos+seq] = k
            self.v_cache[:bsz, :, start_pos:start_pos+seq] = v

            k = self.k_cache[:bsz, :, :start_pos+seq]
            v = self.v_cache[:bsz, :, :start_pos+seq]

        attn_outs = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        attn_outs = attn_outs.contiguous().view(bsz, seq, self.d_model)

        return self.out_proj(attn_outs)


class Block(nn.Module):

    def __init__(self, args: ModelConfig):
        super().__init__()

        self.norm1 = nn.RMSNorm(args.d_model)
        self.gqa = GQA(args)

        self.norm2 = nn.RMSNorm(args.d_model)
        self.swiglu = SwiGLU(args.d_model, args.hidden_dim)

    def forward(self, 
                x: torch.Tensor, 
                rope_embs: torch.Tensor, 
                start_pos: int = 0, 
                use_cache: bool = False):
        x = x + self.gqa(self.norm1(x), rope_embs, start_pos, use_cache)
        x = x + self.swiglu(self.norm2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, args: ModelConfig):
        super().__init__()
        
        self.eos_token_id = args.eos_token_id
        
        self.embeddings = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = nn.RMSNorm(args.d_model)
        self.out_proj = nn.Linear(args.d_model, args.vocab_size, bias=False)
        
        self.register_buffer(
            "rope_embs",
            precompute_rope_embeddings(args.d_model//args.n_heads, args.max_seq_len, args.theta),
            persistent=False
        )

    def forward(self, x: torch.Tensor, start_pos: int = 0, use_cache: bool = False):
        x = self.embeddings(x)

        for block in self.blocks:
            x = block(x, self.rope_embs, start_pos, use_cache)

        return self.out_proj(self.norm(x))

    @torch.inference_mode
    def generate(self, 
                 ids: torch.Tensor,
                 max_tokens: int,
                 temperature: float,
                 top_k: int):
        self.eval() 
        
        bsz, seq_len = ids.shape

        logits = self(ids, start_pos=0, use_cache=True)

        next_logits = logits[:, -1, :] / temperature
        top_values, top_indices = torch.topk(next_logits, k=top_k)
        probs = F.softmax(top_values, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        next_token = top_indices.gather(dim=-1, index=idx)

        ids = torch.cat([ids, next_token], dim=-1)
        start_pos = seq_len
        
        for _ in range(max_tokens):
            logits = self(next_token, start_pos=start_pos, use_cache=True)[:, -1, :]
            logits /= temperature

            top_values, top_indices = torch.topk(logits, k=top_k)
            probs = F.softmax(top_values, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            next_token = top_indices.gather(dim=-1, index=idx)

            ids = torch.cat([ids, next_token], dim=-1)
            start_pos += 1
            
            if next_token.squeeze().item() == self.eos_token_id:
                break

        return ids

