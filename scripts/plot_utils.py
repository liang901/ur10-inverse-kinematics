
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from math import pi

# ====== 繪製損失曲線 ======
def plot_loss(history, save_path):
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Position 2-Norm)')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.show()

def plot_distance_error(df, output_path, bin_width=0.01, show=True):
    errors = df["distance_error"].values

    # ===== 固定誤差門檻下的比例 =====
    thresholds = [0.01, 0.02, 0.05]
    print("\n[Distance Error Threshold Report]")
    for t in thresholds:
        ratio = np.mean(errors < t) * 100
        print(f"  {ratio:6.2f}% of samples have error < {t:.3f}")
    print(f"  Max error: {np.max(errors):.4f}")
    print(f"  Mean error: {np.mean(errors):.4f}")
    print(f"  Std deviation: {np.std(errors):.4f}")
    print("----------------------------------")

    # ===== 畫直方圖 =====
    plt.figure(figsize=(8, 5))
    bins = np.arange(errors.min(), errors.max() + bin_width, bin_width)
    plt.hist(errors, bins=bins, color="skyblue", edgecolor="black")
    plt.xlabel("Distance Error")
    plt.ylabel("Sample Count")
    plt.title("Distance Error Histogram")
    plt.grid(True, linestyle="--", alpha=0.7)

    # x 軸上限：取 min(0.5, 最大誤差 + 5% buffer)
    plt.xlim(0, min(0.5, errors.max() * 1.05))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close()

def plot_distance_error_vectors(df, output_path, show=True):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # ===== 球心 & 半徑 =====
    cx, cy, cz = 0, 0, 0.181
    r = 1.25

    # ===== 畫工作空間球 (wireframe) =====
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    xs = cx + r * np.cos(u) * np.sin(v)
    ys = cy + r * np.sin(u) * np.sin(v)
    zs = cz + r * np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.3, linewidth=0.2)

    # 誤差向量分量
    dx = df["x_pred"] - df["x"]
    dy = df["y_pred"] - df["y"]
    dz = df["z_pred"] - df["z"]

    # 誤差大小 (L2 norm)
    error_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)

    # 誤差正規化
    norm = Normalize(vmin=np.min(error_magnitude), vmax=np.max(error_magnitude))
    cmap = plt.colormaps["coolwarm"]  # 藍 → 紅
    colors = cmap(norm(error_magnitude))  # RGBA (alpha 先為1)

    # 透明度從 0.3 ~ 1.0 映射
    alpha_scaled = norm(error_magnitude) * 0.5 + 0.5

    for xi, yi, zi, dxi, dyi, dzi, rgba, alpha in zip(
        df["x"], df["y"], df["z"], dx, dy, dz, colors, alpha_scaled
    ):
        rgba_with_alpha = list(rgba[:3]) + [alpha]
        error_len = np.sqrt(dxi**2 + dyi**2 + dzi**2)
        ax.quiver(
            xi, yi, zi,
            dxi, dyi, dzi,
            length=error_len,
            normalize=True,
            color=rgba_with_alpha,
            arrow_length_ratio=0.1,
            linewidth=1,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Prediction Error Vectors")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close()

def plot_joint_angle_distribution(df, bin_width = 0.1, show=True):
    joint_columns = [col for col in df.columns if col.startswith("q")]

    for joint in joint_columns:
        angles_wrapped = df[joint].values
        bin_width = bin_width
        bins = np.arange(-pi, pi + bin_width, bin_width)
        plt.figure(figsize=(7, 5))
        plt.hist(angles_wrapped, bins=bins, color="lightgreen", edgecolor="black")
        plt.xlabel("Joint Angle (rad)")
        plt.ylabel("Sample Count")
        plt.title(f"Distribution of {joint}")
        plt.grid(True)

        # 固定範圍
        plt.xlim(-pi, pi)
        plt.ylim(0, 250)

        plt.tight_layout()
        plt.savefig(f"results/{joint}_distribution.png", dpi=300)
        if show:
            plt.show()
        plt.close()

    print(f"[INFO] 已輸出 {len(joint_columns)} 張關節角度分布圖到 results")

