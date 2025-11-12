from typing import final, override

import mlx.core as mx
from mlx import nn

from categorical import Categorical


def apply_rope(x: mx.array, offset: int = 0) -> mx.array:
    """
    Apply Rotary Position Embedding (RoPE) to input tensor.

    Args:
        x: Input tensor of shape [B, T, d_model]
        offset: Position offset (for cached decoding, not used here)

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    *_, seq_len, d_model = x.shape

    # Create position indices
    positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)

    # Create frequency bands (half of d_model since we apply to pairs)
    # Using base 10000 as in original RoPE paper
    freqs = mx.exp(
        mx.arange(0, d_model, 2, dtype=mx.float32)
        * -(mx.log(mx.array(10000.0)) / d_model)
    )

    # Compute angles: [seq_len, d_model//2]
    angles = positions[:, None] * freqs[None, :]

    # Create cos and sin: [seq_len, d_model//2]
    cos = mx.cos(angles)
    sin = mx.sin(angles)

    # Split x into even and odd dimensions: [B, T, d_model//2]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Apply rotation
    # Real part: x_even * cos - x_odd * sin
    # Imaginary part: x_even * sin + x_odd * cos
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # Interleave back: stack and reshape
    # Stack along last dimension: [B, T, d_model//2, 2]
    rotated = mx.stack([rotated_even, rotated_odd], axis=-1)

    # Reshape to [B, T, d_model]
    return rotated.reshape(x.shape)


@final
class PolicyNet(nn.Module):
    def __init__(
        self,
        seq_len: int = 200,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        # input shape: [B, seq_len, 3] - full sequence with attention
        # 3 channels: [my_action, opponent_action, is_episode_start]
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        # Project 3D input (my_action, opponent_action, is_episode_start) to d_model dimensions
        self.input_proj = nn.Linear(3, d_model)

        # Transformer layers - using lists like MLX documentation
        self.attentions = [
            nn.MultiHeadAttention(d_model, n_heads) for _ in range(n_layers)
        ]
        self.ffns = [
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(n_layers)
        ]
        self.ln1s = [nn.LayerNorm(d_model) for _ in range(n_layers)]
        self.ln2s = [nn.LayerNorm(d_model) for _ in range(n_layers)]

        # Output head
        self.output = nn.Linear(d_model, 3)

    @override
    def __call__(
        self,
        obs: mx.array,
        actions: mx.array | None,
    ) -> tuple[Categorical, mx.array | None]:
        policy = self.policy(obs)
        logps = None if actions is None else policy.log_prob(actions)
        return policy, logps

    def policy(self, obs: mx.array) -> Categorical:
        # obs: [B, seq_len, 3]
        _B, T, _ = obs.shape

        # Project input to d_model: [B, T, 3] -> [B, T, d_model]
        x = self.input_proj(obs)

        # Apply RoPE (Rotary Position Embedding)
        x = apply_rope(x)

        # Create causal mask: [T, T] where mask[i,j] = -inf if j > i, else 0
        # This prevents attending to future positions
        causal_mask = mx.triu(mx.full((T, T), -1e9, dtype=mx.float32), k=1)

        # Apply transformer layers
        for i in range(self.n_layers):
            # Self-attention with causal mask
            attn_out = self.attentions[i](x, x, x, mask=causal_mask)
            x = self.ln1s[i](x + attn_out)

            # Feedforward with residual
            ffn_out = self.ffns[i](x)
            x = self.ln2s[i](x + ffn_out)

        # Take the last timestep's representation: [B, T, d_model] -> [B, d_model]
        x = x[:, -1, :]

        # Project to action logits: [B, d_model] -> [B, 3]
        logits = self.output(x)
        return Categorical(logits)


@final
class ValueNet(nn.Module):
    def __init__(
        self,
        seq_len: int = 200,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        # input shape: [B, seq_len, 3] - full sequence with attention
        # 3 channels: [my_action, opponent_action, is_episode_start]
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        # Project 3D input (my_action, opponent_action, is_episode_start) to d_model dimensions
        self.input_proj = nn.Linear(3, d_model)

        # Transformer layers - using lists like MLX documentation
        self.attentions = [
            nn.MultiHeadAttention(d_model, n_heads) for _ in range(n_layers)
        ]
        self.ffns = [
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(n_layers)
        ]
        self.ln1s = [nn.LayerNorm(d_model) for _ in range(n_layers)]
        self.ln2s = [nn.LayerNorm(d_model) for _ in range(n_layers)]

        # Output head
        self.output = nn.Linear(d_model, 1)

    @override
    def __call__(self, obs: mx.array) -> mx.array:
        # obs: [B, seq_len, 3]
        _B, T, _ = obs.shape

        # Project input to d_model: [B, T, 3] -> [B, T, d_model]
        x = self.input_proj(obs)

        # Apply RoPE (Rotary Position Embedding)
        x = apply_rope(x)

        # Create causal mask: [T, T] where mask[i,j] = -inf if j > i, else 0
        # This prevents attending to future positions
        causal_mask = mx.triu(mx.full((T, T), -1e9, dtype=mx.float32), k=1)

        # Apply transformer layers
        for i in range(self.n_layers):
            # Self-attention with causal mask
            attn_out = self.attentions[i](x, x, x, mask=causal_mask)
            x = self.ln1s[i](x + attn_out)

            # Feedforward with residual
            ffn_out = self.ffns[i](x)
            x = self.ln2s[i](x + ffn_out)

        # Take the last timestep's representation: [B, T, d_model] -> [B, d_model]
        x = x[:, -1, :]

        # Project to value: [B, d_model] -> [B, 1]
        return self.output(x)


class EaterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_net = PolicyNet()
        self.v_net = ValueNet()

    def step(self, obs: mx.array) -> tuple[int, float, float]:
        # Add batch dimension if missing
        if obs.ndim == 2:
            obs = mx.expand_dims(obs, axis=0)
        policy = self.p_net.policy(obs)
        action = policy.sample()
        logp = policy.log_prob(action)
        value = self.v_net(obs)
        return (int(action.item()), float(logp.item()), float(value.item()))

    def value(self, obs: mx.array) -> float:
        # Add batch dimension if missing
        if obs.ndim == 2:
            obs = mx.expand_dims(obs, axis=0)
        return float(self.v_net(obs))
