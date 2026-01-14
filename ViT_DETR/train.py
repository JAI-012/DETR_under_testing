import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from my_model import EdgeDETR
from dataset import RobotDataset
from utils.matcher import HungarianMatcher
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EdgeDETR(num_classes=1).to(device)
    ds = RobotDataset('dataset/images', 'dataset/labels')
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    matcher = HungarianMatcher(cost_bbox=5, cost_giou=2)

    model.train()
    print("Starting EdgeDETR training (L1 + GIoU)...")

    for epoch in range(50):
        total_loss = 0.0

        for imgs, targets in loader:
            imgs = torch.stack(imgs).to(device)
            targets = [
                {'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)}
                for t in targets
            ]

            outputs = model(imgs)
            indices = matcher(outputs, targets)

            loss_bbox = 0.0
            loss_giou = 0.0
            n_boxes = 0

            for i, (src_idx, tgt_idx) in enumerate(indices):
                if len(tgt_idx) == 0:
                    continue

                src_boxes = outputs['pred_boxes'][i][src_idx]
                tgt_boxes = targets[i]['boxes'][tgt_idx]

                loss_bbox += F.l1_loss(src_boxes, tgt_boxes, reduction='sum')
                loss_giou += (
                    1 - torch.diag(
                        generalized_box_iou(
                            box_cxcywh_to_xyxy(src_boxes),
                            box_cxcywh_to_xyxy(tgt_boxes)
                        )
                    )
                ).sum()

                n_boxes += len(tgt_idx)

            if n_boxes > 0:
                loss_bbox /= n_boxes
                loss_giou /= n_boxes

                loss = 5.0 * loss_bbox + 2.0 * loss_giou

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), "edge_detr_weights.pth")
    print("Training complete.")

if __name__ == "__main__":
    train()
