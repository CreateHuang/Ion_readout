import torch

import torch.nn as nn

import torch.nn.functional as F


class SiteDIAHead(nn.Module):


    def __init__(

        self,

        num_ions=300,

        in_channels=17,

        hidden_dim=128,

        num_heads=4,

        num_points=4,

        max_attn_offset=4.0,

        residual_attn_offset=1.0,

        psf_sigma=1.5,

        use_psf_guided_offsets=True,

        max_coord_offset=3.0,

        image_size=(88, 456),

        num_ion_attn_layers=1,

    ):

        super().__init__()

        if hidden_dim % num_heads != 0:

            raise ValueError("hidden_dim must be divisible by num_heads")

        if num_points < 1:

            raise ValueError("num_points must be >= 1")

        self.num_ions = num_ions

        self.hidden_dim = hidden_dim

        self.num_heads = num_heads

        self.num_points = num_points

        self.head_dim = hidden_dim // num_heads

        self.max_attn_offset = float(max_attn_offset)

        self.residual_attn_offset = float(residual_attn_offset)

        self.psf_sigma = float(psf_sigma)

        self.use_psf_guided_offsets = bool(use_psf_guided_offsets)

        self.max_coord_offset = float(max_coord_offset)

        self.image_size = image_size


        self.fuse = nn.Sequential(

            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),

            nn.BatchNorm2d(hidden_dim),

            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(hidden_dim),

            nn.ReLU(inplace=True),

        )


        self.ion_embed = nn.Embedding(num_ions, hidden_dim)

        self.coord_embed = nn.Sequential(

            nn.Linear(2, hidden_dim),

            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),

        )


        self.offset_proj = nn.Linear(hidden_dim, num_heads * num_points * 2)

        self.attn_proj = nn.Linear(hidden_dim, num_heads * num_points)

        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(inplace=True),

            nn.LayerNorm(hidden_dim),

        )


        if num_ion_attn_layers > 0:

            self.ion_interaction = nn.TransformerEncoder(

                nn.TransformerEncoderLayer(

                    d_model=hidden_dim,

                    nhead=num_heads,

                    dim_feedforward=hidden_dim * 2,

                    dropout=0.0,

                    batch_first=True,

                    norm_first=True,

                ),

                num_layers=num_ion_attn_layers,

                enable_nested_tensor=False,

            )

        else:

            self.ion_interaction = None


        self.state_head = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, 1),

        )

        self.coord_head = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, 2),

        )

        self.exist_head = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, 1),

        )

        self.uncert_head = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, 1),

        )


        self._init_sampling_bias()


    def _init_sampling_bias(self):


        nn.init.zeros_(self.offset_proj.weight)

        nn.init.zeros_(self.offset_proj.bias)

        nn.init.zeros_(self.attn_proj.weight)

        nn.init.zeros_(self.attn_proj.bias)


    def _normalize_coords(self, coords, height, width):


        x = coords[..., 0] / max(width - 1, 1) * 2.0 - 1.0

        y = coords[..., 1] / max(height - 1, 1) * 2.0 - 1.0

        return torch.stack([x, y], dim=-1)


    def _coords_unit01(self, coords, height, width):

        x = coords[..., 0] / max(width - 1, 1)

        y = coords[..., 1] / max(height - 1, 1)

        return torch.stack([x, y], dim=-1).clamp(0, 1)


    def _build_psf_guided_offsets(self, site_coords):


        B, K, _ = site_coords.shape

        device = site_coords.device

        dtype = site_coords.dtype


        coords32 = site_coords.float()

        offsets = torch.zeros(

            B,

            K,

            self.num_heads,

            self.num_points,

            2,

            device=device,

            dtype=coords32.dtype,

        )

        if K <= 1:

            return offsets


        num_nn = min(self.num_heads, K - 1)

        dist = torch.cdist(coords32, coords32, p=2)

        eye = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)

        dist = dist.masked_fill(eye, float("inf"))

        nn_dist, nn_idx = torch.topk(dist, k=num_nn, dim=-1, largest=False)


        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, K, num_nn)

        neigh = coords32[batch_idx, nn_idx]

        center = coords32.unsqueeze(2)

        vec = neigh - center

        direction = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6)


        if num_nn < self.num_heads:

            pad = self.num_heads - num_nn

            direction = torch.cat([direction, direction[:, :, -1:, :].expand(B, K, pad, 2)], dim=2)

            nn_dist = torch.cat([nn_dist, nn_dist[:, :, -1:].expand(B, K, pad)], dim=2)


        if self.num_points > 1:

            offsets[:, :, :, 1, :] = self.psf_sigma * direction

        if self.num_points > 2:

            offsets[:, :, :, 2, :] = 2.0 * self.psf_sigma * direction

        if self.num_points > 3:

            tail = torch.clamp(0.5 * nn_dist, max=self.max_attn_offset)

            for p in range(3, self.num_points):

                offsets[:, :, :, p, :] = tail.unsqueeze(-1) * direction


        return offsets.clamp(-self.max_attn_offset, self.max_attn_offset).to(dtype=dtype)


    def forward(self, raw_image, feature_map, site_coords):


        B, _, H, W = raw_image.shape

        if site_coords.dim() == 2:

            site_coords = site_coords.unsqueeze(0).expand(B, -1, -1)

        site_coords = site_coords.to(device=raw_image.device, dtype=raw_image.dtype)

        K = site_coords.shape[1]

        if K > self.num_ions:

            raise ValueError(f"Received {K} sites, but head was built for {self.num_ions}")


        if feature_map.shape[-2:] != (H, W):

            feature_map = F.interpolate(feature_map, size=(H, W), mode="bilinear", align_corners=True)

        fused = self.fuse(torch.cat([feature_map, raw_image], dim=1))


        coord01 = self._coords_unit01(site_coords, H, W)

        ids = torch.arange(K, device=raw_image.device)

        query = self.ion_embed(ids).unsqueeze(0).expand(B, -1, -1) + self.coord_embed(coord01)


        residual_offsets = self.residual_attn_offset * torch.tanh(self.offset_proj(query))

        residual_offsets = residual_offsets.view(B, K, self.num_heads, self.num_points, 2)

        if self.use_psf_guided_offsets:

            psf_offsets = self._build_psf_guided_offsets(site_coords)

        else:

            psf_offsets = torch.zeros_like(residual_offsets)


        offsets_px = (psf_offsets + residual_offsets).clamp(

            -self.max_attn_offset, self.max_attn_offset

        )

        attn = self.attn_proj(query).view(B, K, self.num_heads, self.num_points)

        attn = torch.softmax(attn, dim=-1)


        offset_norm_x = offsets_px[..., 0] / max(W - 1, 1) * 2.0

        offset_norm_y = offsets_px[..., 1] / max(H - 1, 1) * 2.0

        offset_norm = torch.stack([offset_norm_x, offset_norm_y], dim=-1)

        ref = self._normalize_coords(site_coords, H, W).view(B, K, 1, 1, 2)

        sample_grid = (ref + offset_norm).clamp(-1.2, 1.2)


        flat_grid = sample_grid.view(B, K * self.num_heads * self.num_points, 1, 2)

        sampled = F.grid_sample(

            fused,

            flat_grid,

            mode="bilinear",

            padding_mode="border",

            align_corners=True,

        )

        sampled = sampled.squeeze(-1).transpose(1, 2)

        sampled = sampled.view(B, K, self.num_heads, self.num_points, self.hidden_dim)

        sampled = self.value_proj(sampled)

        sampled = sampled.view(B, K, self.num_heads, self.num_points, self.num_heads, self.head_dim)


        sampled = torch.stack(

            [sampled[:, :, h, :, h, :] for h in range(self.num_heads)], dim=2

        )


        token = (sampled * attn.unsqueeze(-1)).sum(dim=3).reshape(B, K, self.hidden_dim)

        token = self.out_proj(token + query)


        if self.ion_interaction is not None:

            token = self.ion_interaction(token)


        bright_logit = self.state_head(token).squeeze(-1)

        coord_delta = self.max_coord_offset * torch.tanh(self.coord_head(token))

        pred_coords = site_coords + coord_delta

        exist_logit = self.exist_head(token).squeeze(-1)

        uncertainty = F.softplus(self.uncert_head(token).squeeze(-1))


        return {

            "tokens": token,

            "bright_logit": bright_logit,

            "pred_coords": pred_coords,

            "exist_logit": exist_logit,

            "uncertainty": uncertainty,

            "dia_offsets": offsets_px,

            "dia_psf_offsets": psf_offsets,

            "dia_residual_offsets": residual_offsets,

            "dia_attn": attn,

        }

