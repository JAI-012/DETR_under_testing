import torch
import torch.nn as nn
import torchvision.models as models

class EdgeDETR(nn.Module):
    def __init__(self, num_classes=1, hidden_dim=256, nheads=8, num_enc=4, num_dec=4):
        super().__init__()
        # 1. Backbone: MobileNetV3 (Lightweight for Jetson)
        self.backbone = models.mobilenet_v3_large(pretrained=True).features
        self.conv = nn.Conv2d(960, hidden_dim, 1) # Project to 256 dim

        # 2. Transformer (The Brain)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=nheads,
            num_encoder_layers=num_enc, num_decoder_layers=num_dec
        )

        # 3. Heads
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1) # +1 for Background
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim)) # 100 Guesses
        self.row_embed = nn.Parameter(torch.rand(150, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(150, hidden_dim // 2))

    def forward(self, x):
        # x shape: [batch, 3, H, W]
        feat = self.conv(self.backbone(x))
        H, W = feat.shape[-2:]
        
        # Positional Embeddings
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # Transformer Pass
        h = self.transformer(pos + feat.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1).repeat(1, x.shape[0], 1))
        
        return {
            'pred_logits': self.linear_class(h.transpose(0, 1)),
            'pred_boxes': self.linear_bbox(h.transpose(0, 1)).sigmoid()
        }