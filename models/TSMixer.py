import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, seq_len, latent_dim, expansion_factor=2, dropout=0.1):
        super().__init__()
        hidden_time = int(seq_len * expansion_factor)
        hidden_feat = int(latent_dim * expansion_factor)
        self.layernorm_time = nn.LayerNorm(latent_dim)
        self.layernorm_feat = nn.LayerNorm(latent_dim)
        self.time_mlp = MLPBlock(seq_len, hidden_time, dropout)
        self.feature_mlp = MLPBlock(latent_dim, hidden_feat, dropout)

    def forward(self, x):
        # x: (B, L, latent_dim)
        residual = x
        x_t = self.layernorm_time(x)
        x_t = x_t.transpose(1, 2)      # (B, latent_dim, L)
        x_t = self.time_mlp(x_t)
        x_t = x_t.transpose(1, 2)      # (B, L, latent_dim)
        x = residual + x_t

        residual = x
        x_f = self.layernorm_feat(x)
        x_f = self.feature_mlp(x_f)
        x = residual + x_f
        return x

class TSMixer(nn.Module):
    def __init__(self, input_dim=None, pred_len=None, latent_dim=64, expansion_factor=2, dropout=0.1, args=None, n_blocks=4, ):
        super().__init__()

        # 🧩 Support both direct numbers or Namespace config
        if args is not None:
            seq_len = getattr(args, "seq_len", 96)
            pred_len = getattr(args, "pred_len", 96)
            input_dim = getattr(args, "enc_in", input_dim)
        else:
            seq_len = pred_len  # default fallback if not specified

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.n_blocks = n_blocks

        hidden_time = int(seq_len * expansion_factor)
        hidden_feat = int(latent_dim * expansion_factor)

        self.blocks = nn.ModuleList([
            MixerBlock(seq_len, latent_dim, expansion_factor, dropout)
            for _ in range(4)
        ])
        self.fc_out = nn.Linear(latent_dim, input_dim)
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 🔹 Feature adapters
        self.input_adapter = nn.Linear(input_dim, latent_dim)
        self.output_adapter = nn.Linear(latent_dim, input_dim)

        # 🔹 Core mixer blocks
        self.blocks = nn.ModuleList([
            MixerBlock(seq_len, latent_dim, expansion_factor, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        x = self.input_adapter(x)       # map to latent
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)          # (B, latent_dim, L)
        x = self.proj(x)                # (B, latent_dim, pred_len)
        x = x.permute(0, 2, 1)          # (B, pred_len, latent_dim)
        x = self.output_adapter(x)      # map back to input_dim
        return x
