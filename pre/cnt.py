import cv2
import os
from glob import glob

# ===== 裁剪范围定义 =====
crop_top = 28
crop_bottom = 53
crop_left = 35
crop_right = 35

# ===== 根目录 =====
root_dir = "../data"

# ===== 递归搜索所有 .tiff / .bmp 文件 =====
image_files = (
    glob(os.path.join(root_dir, "**", "*.tiff"), recursive=True) +
    glob(os.path.join(root_dir, "**", "*.bmp"), recursive=True)
)

print(f"共找到 {len(image_files)} 个文件")

# ===== 遍历所有文件 =====
for file in sorted(image_files):
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"无法读取: {file}")
        continue

    h, w = img.shape[:2]
    y1, y2 = crop_top, h - crop_bottom
    x1, x2 = crop_left, w - crop_right

    # 防止裁剪范围越界
    if y2 <= y1 or x2 <= x1:
        print(f"裁剪范围非法: {file} ({w}×{h})")
        continue

    cropped = img[y1:y2, x1:x2]

    cv2.imwrite(file, cropped)
    print(f"已裁剪: {file}")

print("裁剪完成")
