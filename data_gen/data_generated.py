import os
import numpy as np
import pandas as pd

# ====== 種子碼(確保可重現) ======
SEED = 42
np.random.seed(SEED)

# ====== 生成資料 ======
def generate_positions(num_samples, radius=1.25, center=(0.0, 0.0, 0.181)):
    """
    在指定球體範圍內隨機生成 (x, y, z) 坐標
    :param num_samples: 要生成的樣本數
    :param radius: 球半徑 (m)
    :param center: 球心座標 (x, y, z) (m)
    :return: numpy array (num_samples, 3)
    """
    points = []
    cx, cy, cz = center

    while len(points) < num_samples:
        # 在 [-r, r] 的立方體中隨機取點
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        z = np.random.uniform(-radius, radius)

        # 篩選落在球內的點
        if x**2 + y**2 + z**2 <= radius**2:
            points.append((cx + x, cy + y, cz + z))

    return np.array(points, dtype=np.float32)

# ====== 主程式 ======
if __name__ == "__main__":
    NUM_SAMPLES = 15000  # 樣本數
    os.makedirs("data", exist_ok=True)
    save_path = "data/positions_dataset.csv"

    positions = generate_positions(NUM_SAMPLES)
    df = pd.DataFrame(positions, columns=["x", "y", "z"])
    df.to_csv(save_path, index=True)

    print(f"已生成 {NUM_SAMPLES} 筆資料，儲存至 {save_path}")
