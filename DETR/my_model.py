import torch
import torch.nn as nn
import torchvision.models as models

class EdgeDETR(nn.Module):
    def __init__(self, num_classes=1, hidden_dim=256, nheads=8, num_enc=4, num_dec=4):
        super().__init__()
        # 1. Backbone: MobileNetV3 (Lightweight for Jetson)
        self.backbone = models.mobilenet_v3_large(pretrained=True).features
        self.conv = nn.Conv2d(960, hidden_dim, 1) # Project to 256 dim
        self.hidden_dim = hidden_dim

        # 2. Transformer (The Brain)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=nheads,
            num_encoder_layers=num_enc, num_decoder_layers=num_dec
        )

        # 3. Heads
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1) # +1 for Background
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim)) # 100 Guesses

    def _get_positional_encoding(self, H, W, num_pos_feats):
        #Generate 2D sinusoidal positional encoding
        y_embed = torch.arange(H, dtype=torch.float32).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32).unsqueeze(0).repeat(H, 1)
        
        y_embed = y_embed / H
        x_embed = x_embed / W
        
        dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (num_pos_feats // 2))
        
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)
        
        pos = torch.cat([pos_y, pos_x], dim=2)
        return pos.flatten(0, 1).unsqueeze(1)

    def forward(self, x):
        # x shape: [batch, 3, H, W]
        feat = self.conv(self.backbone(x))
        H, W = feat.shape[-2:]
        
        # Positional Embeddings
        pos = self._get_positional_encoding(H, W, self.hidden_dim).to(x.device)

        # Transformer Pass
        h = self.transformer(pos + feat.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1).repeat(1, x.shape[0], 1))
        
        return {
            'pred_logits': self.linear_class(h.transpose(0, 1)),
            'pred_boxes': self.linear_bbox(h.transpose(0, 1)).sigmoid()
        }