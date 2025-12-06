import cv2
import numpy as np
import os
from glob import glob

# ===== 根目录 =====
root_dir = "../raw_data"
eps = 1e-6
to_save_dir = "../data"

image_files = sorted(glob(os.path.join(root_dir, "*.bmp")))
print(f"找到 {len(image_files)} 张图像")

# ===== 逐张处理 =====
for file in image_files:
    base_name = os.path.splitext(os.path.basename(file))[0]
    save_dir = os.path.join(to_save_dir, base_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"正在处理: {file}")

    # ===== 读取原始图像 =====
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32)
    h, w = img.shape

    # ===== 解码四个偏振方向 =====
    I0 = img[0:h:2, 0:w:2]
    I45 = img[0:h:2, 1:w:2]
    I90 = img[1:h:2, 0:w:2]
    I135 = img[1:h:2, 1:w:2]

    # ===== 强度图 =====
    Intensity = (I0 + I90) / 2.0
    cv2.imwrite(os.path.join(save_dir, "Intensity.bmp"), Intensity.astype(np.uint8))

    # ===== 偏振度图 DoLP =====
    DoLP = np.sqrt((I0 - I90) ** 2 + (I45 - I135) ** 2) / (I0 + I90 + eps)
    DoLP = np.clip(DoLP, 0, 1)
    cv2.imwrite(os.path.join(save_dir, "DoLP.tiff"), (DoLP * 65535).astype(np.uint16))

    # ===== 偏振角图 AoLP' =====
    AoLP = 0.5 * np.arctan2((I45 - I135), (I0 - I90))
    AoLP_sin = np.sin(2 * AoLP)
    AoLP_cos = np.cos(2 * AoLP)

    cv2.imwrite(os.path.join(save_dir, "AoLP_sin.tiff"), ((AoLP_sin + 1.0) * 32767.5).astype(np.uint16))
    cv2.imwrite(os.path.join(save_dir, "AoLP_cos.tiff"), ((AoLP_cos + 1.0) * 32767.5).astype(np.uint16))

print("处理完成！")
