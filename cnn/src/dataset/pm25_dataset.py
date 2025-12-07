import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class PM25Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data")

        import pandas as pd
        df = pd.read_csv(os.path.join(root_dir, ""), header=None, names=["folder", "pm25"])
        self.label_dict = {row.folder: float(row.pm25) for _, row in df.iterrows()}

        self.folders = sorted(
            [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))],
            key=lambda x: int("".join(filter(str.isdigit, x)))
        )


    def read_gray_tiff(self, path, div):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return img.astype(np.float32) / div

    def __getitem__(self, idx):
        folder = self.folders[idx]
        fp = os.path.join(self.data_dir, folder)

        # ========== 读取图像 ==========
        inten = self.read_gray_tiff(os.path.join(fp, "Intensity.bmp"), 255)
        dolp = self.read_gray_tiff(os.path.join(fp, "DoLP.tiff"), 65535)
        a_s = self.read_gray_tiff(os.path.join(fp, "AoLP_sin.tiff"), 65535)
        a_c = self.read_gray_tiff(os.path.join(fp, "AoLP_cos.tiff"), 65535)

        # ========== AoLP 范围恢复 (-1 ~ 1) ==========
        a_s = a_s * 2 - 1
        a_c = a_c * 2 - 1

        # ========== 堆叠成 4 通道 ==========
        img = np.stack([inten, dolp, a_s, a_c], axis=0).astype(np.float32)
        img = torch.from_numpy(img)

        # ========== Padding + Resize ==========
        H, W = img.shape[1:]
        M = max(H, W)
        pad_h = (M - H) // 2
        pad_w = (M - W) // 2

        img = F.pad(
            img,
            (pad_w, M - W - pad_w, pad_h, M - H - pad_h),
            mode='constant',
            value=0
        )

        img = TF.resize(img, [224, 224])

        label = torch.tensor(self.label_dict[folder], dtype=torch.float32)
        return img, label

    def __len__(self):
        return len(self.folders)
