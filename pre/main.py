import cv2
import numpy as np
import os
from glob import glob

# ===== 根目录 =====
root_dir = "./output"

eps = 1e-6  # 防止除零

# ===== 遍历所有子文件夹 =====
for folder in sorted(glob(os.path.join(root_dir, "*"))):
    if not os.path.isdir(folder):
        continue

    print(f"📂 正在处理文件夹: {folder}")

    # 匹配 .tiff / .tif 文件
    image_files = glob(os.path.join(folder, "*.tiff")) + glob(os.path.join(folder, "*.tif"))
    if not image_files:
        print(f"⚠️ 未找到图像文件: {folder}")
        continue

    for file in sorted(image_files):
        base_name = os.path.splitext(os.path.basename(file))[0]
        save_dir = os.path.join(folder, base_name)
        os.makedirs(save_dir, exist_ok=True)

        # ====== 读取原始图像 ======
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"❌ 无法读取图像: {file}")
            continue

        img = img.astype(np.float32)
        h, w = img.shape

        # ====== 解码四个偏振方向 ======
        I0   = img[0:h:2, 0:w:2]
        I45  = img[0:h:2, 1:w:2]
        I90  = img[1:h:2, 0:w:2]
        I135 = img[1:h:2, 1:w:2]

        # ====== 强度图 ======
        Intensity = (I0 + I90) / 2.0
        cv2.imwrite(os.path.join(save_dir, "Intensity.tiff"), Intensity.astype(np.uint16))

        # ====== 偏振度图 (DoLP) ======
        DoLP = np.sqrt((I0 - I90)**2 + (I45 - I135)**2) / (I0 + I90 + eps)
        DoLP = np.clip(DoLP, 0, 1)
        cv2.imwrite(os.path.join(save_dir, "DoLP.tiff"), (DoLP * 65535).astype(np.uint16))

        # ====== 偏振角图 (AoLP) ======
        AoLP = 0.5 * np.arctan2((I45 - I135), (I0 - I90))  # [-π/2, π/2]
        AoLP_norm = (AoLP + np.pi/2) / np.pi               # 归一化到 [0,1]
        cv2.imwrite(os.path.join(save_dir, "AoLP.tiff"), (AoLP_norm * 65535).astype(np.uint16))

        # ====== 删除原图像 ======
        try:
            os.remove(file)
        except Exception as e:
            print(f"⚠️ 删除失败: {file}, 原因: {e}")

        print(f"✅ 处理完成: {file} → {save_dir}")