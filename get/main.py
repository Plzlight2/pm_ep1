import sys
sys.path.append(r"C:\Users\Plzlight2\Desktop\VimbaPython\Source")
import time
import cv2
import os
import numpy as np
from datetime import datetime, timedelta
from vimba import *

# 读 csv
def read_last_line(csv_path):
    with open(csv_path, "rb") as f:
        f.seek(0, 2)
        pos = f.tell()

        # 跳过最后的空行
        while pos > 0:
            pos -= 1
            f.seek(pos)
            if f.read(1) == b'\n':  # 找到换行
                # 看下一行是否为空
                line = f.readline().decode().strip()
                if line != "":     # 找到非空行
                    return line
                # 如果是空行，继续往前搜索
        return None

def get_pm25_from_csv(csv_path):
    last = read_last_line(csv_path)
    parts = last.split(',')
    if len(parts) < 3:
        return None
    return float(parts[2])  # 第三列 PM2.5

def write_label(sample_name, pm25_avg, save_dir):
    label_path = os.path.join(save_dir, "label.csv")
    with open(label_path, "a", encoding="utf-8") as f:
        f.write(f"{sample_name},{pm25_avg:.1f}\n")


# 拍照
def capture_images(duration_hours, interval_sec, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    csv1 = "pm_log1.csv"
    csv2 = "pm_log2.csv"

    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        if not cams:
            raise RuntimeError("未检测到相机")

        with cams[0] as cam:
            print("使用相机:", cam.get_name())

            # === 设置像素格式 ===
            cam.set_pixel_format(PixelFormat.Mono8)

            # === 曝光 & 触发 ===
            cam.ExposureTimeAbs.set(100000.0)
            cam.TriggerMode.set("Off")

            # === 开始循环采集 ===
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=duration_hours)

            print(f"开始采集，共运行 {duration_hours} 小时，每隔 {interval_sec}s 拍摄一张。")

            next_time = time.time()
            counter = 1
            while datetime.now() < end_time:
                frame = cam.get_frame(timeout_ms=2000)
                np_img = frame.as_numpy_ndarray()

                sample_name = f"sample{counter}"
                filename = os.path.join(save_dir, f"{sample_name}.bmp")

                cv2.imwrite(filename, np_img.astype(np.uint8))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 已保存 {filename}")

                # ------------- 读取 2 个 CSV 最新 PM2.5 -------------
                pm25_1 = get_pm25_from_csv(csv1)
                pm25_2 = get_pm25_from_csv(csv2)

                if pm25_1 is not None and pm25_2 is not None:
                    pm25_avg = (pm25_1 + pm25_2) / 2.0
                    write_label(sample_name, pm25_avg, save_dir)
                    print(f"记录 {sample_name} → PM2.5 平均值: {pm25_avg:.3f}")
                else:
                    print("无法读取 CSV 最新数据")

                counter += 1

                next_time += interval_sec
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            print(f"采集完成")

if __name__ == "__main__":
    capture_images(4, 2, r"D:\data_temp")