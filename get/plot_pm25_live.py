import pandas as pd
import matplotlib.pyplot as plt
import time

columns = ["time","pm1","pm25","pm10","n03","n05","n10","n25","n50","n100"]

plt.ion()
fig, ax = plt.subplots(figsize=(10,6))

WINDOW = 180   # 最近180秒

while True:
    try:
        # === 读取两个传感器数据 ===
        df1 = pd.read_csv("pm_log1.csv", header=None, names=columns)
        df2 = pd.read_csv("pm_log2.csv", header=None, names=columns)

        # === 时间戳转换 ===
        df1["time"] = pd.to_datetime(df1["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        df2["time"] = pd.to_datetime(df2["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")

        # 丢弃异常行
        df1 = df1.dropna()
        df2 = df2.dropna()

        if len(df1) > 0 and len(df2) > 0:
            # === 统一时间轴 ===
            t0 = min(df1["time"].iloc[0], df2["time"].iloc[0])
            df1["t"] = (df1["time"] - t0).dt.total_seconds()
            df2["t"] = (df2["time"] - t0).dt.total_seconds()

            # 只保留最近窗口
            t_now = max(df1["t"].max(), df2["t"].max())
            df1 = df1[df1["t"] >= t_now - WINDOW]
            df2 = df2[df2["t"] >= t_now - WINDOW]

            # === 平滑 ===
            df1["pm25_smooth"] = df1["pm25"].rolling(window=1, min_periods=1).mean()
            df2["pm25_smooth"] = df2["pm25"].rolling(window=1, min_periods=1).mean()

            # === 时间对齐融合 ===
            merged = pd.merge_asof(
                df1[["time","pm25_smooth"]],
                df2[["time","pm25_smooth"]],
                on="time",
                direction="nearest",
                tolerance=pd.Timedelta("1s"),
                suffixes=("_1","_2")
            )
            merged["pm25_merged"] = merged[["pm25_smooth_1", "pm25_smooth_2"]].mean(axis=1, skipna=True)
            merged["pm25_final"] = merged["pm25_merged"].rolling(window=1, min_periods=1).mean()
            merged["t"] = (merged["time"] - t0).dt.total_seconds()
            merged = merged[merged["t"] >= t_now - WINDOW]

            # === 绘图 ===
            ax.clear()
            ax.plot(df1["t"], df1["pm25_smooth"], label="Sensor 1 (COMx)", alpha=0.5)
            ax.plot(df2["t"], df2["pm25_smooth"], label="Sensor 2 (COMy)", alpha=0.5)
            ax.plot(merged["t"], merged["pm25_final"], label="Fused Avg", linewidth=2.0, color="black")

            ax.set_xlim(t_now - WINDOW, t_now)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("PM2.5 (µg/m³)")
            ax.set_title("PM2.5 Real-Time Fusion (Smoothed, Last 180s)")
            ax.legend()
            ax.grid(True)

            # === 最新值与相对误差 ===
            latest1 = df1["pm25"].iloc[-1]
            latest2 = df2["pm25"].iloc[-1]
            latest_fused = merged["pm25_final"].iloc[-1]

            rel_error = abs(latest1 - latest2) / max(latest1, latest2) if max(latest1, latest2) > 0 else float("inf")

            # === 文本显示 ===
            text = (
                f"Sensor1 = {latest1:.1f} µg/m³\n"
                f"Sensor2 = {latest2:.1f} µg/m³\n"
                f"Fused = {latest_fused:.1f} µg/m³\n"
                f"Rel Error = {rel_error*100:.1f}%"
            )

            ax.text(
                0.95, 0.95, text,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )

        plt.draw()
        plt.pause(1)

    except Exception as e:
        print("等待数据中...", e)
        time.sleep(1)