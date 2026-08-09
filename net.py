from dataclasses import replace
import math
import numpy as np
import torch
import torch.nn as nn

from common_net.attentions import CrossMAB, LinearAttention, MHAConfig
from common_net.common import MLP, ZeroCenteredRMSNorm

def fourier_encode_xy(xy: torch.Tensor, num_frequencies: int=8) -> torch.Tensor:
    freqs = 2 ** torch.arange(num_frequencies, device=xy.device).float()
    x_ = xy.unsqueeze(-1) * freqs  # (..., 2, num_frequencies)
    enc = torch.cat([torch.sin(x_), torch.cos(x_)], dim=-1)  # (..., 2, 2*num_frequencies)

    return enc.flatten(-2)  # (..., 4*num_frequencies)

def fourier_encode_R(R: torch.Tensor, num_frequencies: int=8) -> torch.Tensor:
    r_log = torch.log1p(R) # compress the range of R
    freqs = 2 ** torch.arange(num_frequencies, device=R.device).float()
    # (..., 1, num_frequencies)
    r_ = r_log.unsqueeze(-1) * freqs
    enc = torch.cat([torch.sin(r_), torch.cos(r_)], dim=-1)  # (..., 2*num_frequencies)
    return enc.flatten(-2)  # (..., 2*num_frequencies)

class TargetEncoder(nn.Module):
    def __init__(self,
        embed_dim: int = 64,
        history_len: int = 8,
        num_freqs: int = 8
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.history_len = history_len
        self.num_freqs = num_freqs

        xy_fourier_dim = 4 * num_freqs
        step_input_dim = xy_fourier_dim + 1

        self.step_proj = nn.Sequential(
            nn.Linear(step_input_dim, embed_dim),
            nn.GELU(),
        )

        self.time_pos_emb = nn.Embedding(history_len, embed_dim)

        self.temporal = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.0
        )


    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        # targets: (B, T, H, 3)

        B, T, H, _ = targets.shape

        xy = targets[..., :2]  # (B, T, H, 2)
        flag = targets[..., 2].unsqueeze(-1)  # (B, T, H, 1)

        xy_fourier = fourier_encode_xy(xy, self.num_freqs)  # (B, T, H, xy_fourier_dim)


        xy_enc = xy_fourier * flag

        step_input = torch.cat([xy_enc, flag], dim=-1)  # (B, T, H, step_input_dim)

        tokens: torch.Tensor = self.step_proj(step_input)  # (B, T, H, embed_dim)



        t_idx = torch.arange(H, device=targets.device) # (H,)



        tokens = tokens + self.time_pos_emb(t_idx)  # (B, T, H, embed_dim)

        tokens = tokens.reshape(B*T, H, self.embed_dim)  # (B*T, H, embed_dim) GRU expects (B, seq_len, input_size)

        tokens, _ = self.temporal(tokens)  # (B*T, H, embed_dim)

        tokens = tokens.reshape(B, T, H, self.embed_dim)  # (B, T, H, embed_dim)

        token_vis = tokens * flag  # (B, T, H, embed_dim)


        vis_count = flag.sum(dim=-2).clamp(min=1)  # (B, T, 1)

        pooled = token_vis.sum(dim=-2) / vis_count  # (B, T, embed_dim)

        return pooled  # (B, T, embed_dim)

class CameraEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.dynamic_encoder = nn.Sequential(

        )


class GoalsGenerator(nn.Module):
    def __init__(
        self, embed_dim: int = 64, multi_headed: bool = False, num_heads: int = 2
    ):
        super().__init__()

        # NOT USED for now
        self.multi_headed = multi_headed
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads


        self.q_proj = MLP(input_dim=embed_dim, output_dim=embed_dim, bias=False)
        self.k_proj = MLP(input_dim=embed_dim, output_dim=embed_dim, bias=False)

        self.q_norm = ZeroCenteredRMSNorm(embed_dim)
        self.k_norm = ZeroCenteredRMSNorm(embed_dim)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        # Q: (B, C, D)
        # K: (B, T, D)

        D = Q.shape[-1]


        Q_proj: torch.Tensor = self.q_proj(Q)  # (B, C, D)
        K_proj: torch.Tensor = self.k_proj(K)  # (B, T, D)

        Q_proj = self.q_norm(Q_proj)  # (B, C, D)
        K_proj = self.k_norm(K_proj)  # (B, T, D)

        QKT = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(D)  # (B, C, T)
        # QKT = torch.softmax(QKT, dim=-1, dtype=torch.float32).to(Q.dtype)  # (B, C, T)

        return QKT  # (B, C, T)


class NormalEncoderBlock(nn.Module):
    pass


class ExpandedEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        mha_config: MHAConfig = MHAConfig(),
    ):
        super().__init__()

        self.target_attn = CrossMAB(
            embed_dim, d_ff=embed_dim * 4, mha_config=mha_config
        )

        self.obstacle_attn = CrossMAB(
            embed_dim, d_ff=embed_dim * 4, mha_config=mha_config
        )

        self.warehouse_attn = CrossMAB(
            embed_dim, d_ff=embed_dim * 4, mha_config=mha_config
        )

        self.concat_mlp = MLP(input_dim=embed_dim * 2, output_dim=embed_dim)

    def forward(
        self,
        cam_emb: torch.Tensor,
        obs_emb: torch.Tensor,
        tar_emb: torch.Tensor,
        wh_emb: torch.Tensor,
    ) -> torch.Tensor:
        # cam_emb: C x embed_dim
        # obs_emb: O x embed_dim
        # tar_emb: T x embed_dim
        # wh_emb: W x embed_dim

        tar_attn_out = self.target_attn(cam_emb, tar_emb)  # C x embed_dim
        obs_attn_out = self.obstacle_attn(tar_attn_out, obs_emb)  # C x embed_dim
        wh_attn_out = self.warehouse_attn(tar_attn_out, wh_emb)  # C x embed_dim

        concat_attn_out = torch.cat(
            [obs_attn_out, wh_attn_out], dim=-1
        )  # C x (2*embed_dim)

        out = self.concat_mlp(concat_attn_out)  # C x embed_dim

        return out


class EncoderNet(nn.Module):
    def __init__(
        self,
        n_linear_attn_blocks: int = 0,
        n_attn_blocks: int = 3,
        embed_dim: int = 64,
        mha_config: MHAConfig = MHAConfig(),
    ):
        super().__init__()

        linear_mha_config = replace(mha_config, attn_cls=LinearAttention)

        self.blocks = nn.ModuleList(
            [
                ExpandedEncoderBlock(embed_dim=embed_dim, mha_config=linear_mha_config)
                for _ in range(n_linear_attn_blocks)
            ]
            + [
                ExpandedEncoderBlock(embed_dim=embed_dim, mha_config=mha_config)
                for _ in range(n_attn_blocks - n_linear_attn_blocks)
            ]
        )

    def forward(
        self,
        cameras: torch.Tensor,
        targets: torch.Tensor,
        obstacles: torch.Tensor,
        warehouses: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            cam_emb = block(cameras, targets, obstacles, warehouses)  # C x embed_dim

        return cam_emb

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        history_len: int = 8,
        camera_dim: int = 8,
        target_dim: int = 3,  # default of 8 history
        n_linear_attn_blocks: int = 0,
        n_attn_blocks: int = 3,
        embed_dim: int = 128,
        xy_num_fourier: int = 8,
        R_num_fourier: int = 8,
        mha_config: MHAConfig = MHAConfig(num_heads=1),
    ):
        super().__init__()

        xy_dim = 4 * xy_num_fourier
        R_dim = 2 * R_num_fourier

        self.xy_num_fourier = xy_num_fourier
        self.R_num_fourier = R_num_fourier

        self.cameras = MLP(input_dim=camera_dim*history_len, output_dim=embed_dim, hidden_dims=[embed_dim, embed_dim], bias=False)

        self.targets = TargetEncoder(embed_dim=embed_dim, history_len=history_len, num_freqs=xy_num_fourier)  # target_dim//3 because each step has 3 values (x, y, flag)

        self.obstacles = MLP(
            input_dim=xy_dim + R_dim, output_dim=embed_dim, hidden_dims=[embed_dim, embed_dim]
        )
        self.warehouses = MLP(
            input_dim=xy_dim, output_dim=embed_dim, hidden_dims=[embed_dim, embed_dim]
        )

        self.encoder = EncoderNet(
            n_linear_attn_blocks=n_linear_attn_blocks,
            n_attn_blocks=n_attn_blocks,
            embed_dim=embed_dim,
            mha_config=mha_config,
        )  # from C x embed_dim, T x embed_dim, O x embed_dim, W x embed_dim -> C x embed_dim



    def forward(
        self,
        X: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # cameras: B x C x H x camera_dim
        # Targets: B x T x G x target_dim
        # obstacles: B x O x obstacle_dim
        # warehouses: B x W x warehouse_dim

        warehouse_fourier = fourier_encode_xy(X["warehouses"], self.xy_num_fourier)  # B x W x xy_fourier_dim
        obstacle_fourier_xy = fourier_encode_xy(X["obstacles"][..., :2], self.xy_num_fourier)  # B x O x xy_fourier_dim
        obstacle_fourier_R = fourier_encode_R(X["obstacles"][..., 2].unsqueeze(-1), self.R_num_fourier)  # B x O x R_fourier_dim
        obstacle_fourier = torch.cat([obstacle_fourier_xy, obstacle_fourier_R], dim=-1)  # B x O x (xy_fourier_dim + R_fourier_dim)

        B, C, H, cam_dim = X["cameras"].shape

        cam_emb = self.cameras(X["cameras"].reshape(B, C, -1))  # B x C x embed_dim
        tar_emb = self.targets(X["targets"])  # B x T x embed_dim
        obs_emb = self.obstacles(obstacle_fourier)  # B x O x embed_dim
        wh_emb = self.warehouses(warehouse_fourier)  # B x W x embed_dim

        out = self.encoder(cam_emb, tar_emb, obs_emb, wh_emb)  # B x C x embed_dim

        return out

class Net(nn.Module):
    def __init__(
        self,
        history_len: int = 8,
        camera_dim: int = 8,  # default of 8 history
        target_dim: int = 3,  # default of 8 history
        n_linear_attn_blocks: int = 0,
        n_attn_blocks: int = 3,
        embed_dim: int = 64,
        mha_config: MHAConfig = MHAConfig(num_heads=1),
        xy_num_fourier: int = 8,
        R_num_fourier: int = 8,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            history_len=history_len,
            camera_dim=camera_dim,
            target_dim=target_dim,
            n_linear_attn_blocks=n_linear_attn_blocks,
            n_attn_blocks=n_attn_blocks,
            embed_dim=embed_dim,
            mha_config=mha_config,
            xy_num_fourier=xy_num_fourier,
            R_num_fourier=R_num_fourier,

        )

        self.targets = TargetEncoder(embed_dim=embed_dim, history_len=history_len, num_freqs=xy_num_fourier)

        self.goals_generator = GoalsGenerator(
            embed_dim=embed_dim
        )  # from C x embed_dim, T x embed_dim -> C x T

    def forward(
        self,
        X: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # cameras: B x C x camera_dim
        # Targets: B x T x target_dim
        # obstacles: B x O x obstacle_dim
        # warehouses: B x W x warehouse_dim

        features = self.feature_extractor(X)  # B x C x embed_dim

        tar_emb = self.targets(X['targets'])  # B x T x embed_dim

        goals = self.goals_generator(features, tar_emb)  # B x C x T

        return goals


def main():
    test_mlp = Net()

    temp_input_data = {
        "cameras": torch.randn(1, 4, 8, 8),  # cameras 1, 4, 8, 8
        "targets": torch.randn(1, 12, 8, 3),  # targets 1, 12, 8, 3
        "obstacles": torch.randn(1, 9, 3),  # obstacles 1, 9, 3
        "warehouses": torch.randn(1, 4, 2),  # warehouses 1, 4, 2
    }

    temp_input_data["obstacles"][..., 2] = abs(temp_input_data["obstacles"][..., 2])    # ensure R is non-negative

    # summary(test_mlp, input_data=temp_input_data, col_names=["input_size", "output_size", "num_params"], depth=5)

    print(test_mlp(temp_input_data))  # 1 x C x T

    print((test_mlp(temp_input_data)>0).sum()/(4*12))  # check sparsity of goals


if __name__ == "__main__":
    main()
