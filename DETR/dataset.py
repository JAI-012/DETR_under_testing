import torch
from torch.utils.data import Dataset
import cv2
import os

class RobotDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.img_dir = img_dir
        self.label_dir = label_dir

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        # Load Image
        img_name = self.img_files[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Load Label
        label_file = img_name.replace('.jpg', '.txt')
        boxes, labels = [], []
        if os.path.exists(os.path.join(self.label_dir, label_file)):
            with open(os.path.join(self.label_dir, label_file)) as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    labels.append(int(vals[0])) 
                    boxes.append(vals[1:]) # cx, cy, w, h

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        return img_tensor, target