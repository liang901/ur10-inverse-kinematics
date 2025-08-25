import numpy as np
import pandas as pd
import tensorflow as tf
from model_utils import forward_kinematics, position_loss_mse, position_loss_mae, position_loss_distance
from plot_utils import plot_distance_error, plot_distance_error_vectors, plot_joint_angle_distribution

# ====== 評估模型多種指標 ======
def evaluate_model_metrics(model, X_test):
    y_pred = model.predict(X_test)
    pos_true = tf.constant(X_test, dtype=tf.float32)
    joints_pred = tf.constant(y_pred, dtype=tf.float32)

    # Reuse existing loss functions
    mse = position_loss_mse(pos_true, joints_pred).numpy()
    mae = position_loss_mae(pos_true, joints_pred).numpy()
    mean_2norm = position_loss_distance(pos_true, joints_pred).numpy()

    # R^2 score
    _, pos_pred = forward_kinematics(joints_pred)
    ss_res = tf.reduce_sum(tf.square(pos_pred - pos_true))
    ss_tot = tf.reduce_sum(tf.square(pos_true - tf.reduce_mean(pos_true, axis=0)))
    r2 = (1 - ss_res / ss_tot).numpy()

    print(f"Test Metrics:")
    print(f"Mean 2-Norm Distance: {mean_2norm:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"R^2: {r2:.6f}")

    return mean_2norm, mae, mse, r2

# Wrap angles to [-pi, pi]; if using tanh, apply after scaling the output
def wrap_to_pi(angle_array):
    return (angle_array + np.pi) % (2 * np.pi) - np.pi

def save_test_predictions(model, pos_true, save_path):
    joints_pred = model.predict(pos_true)
    joints_pred = wrap_to_pi(joints_pred)

    pos_pred = forward_kinematics(tf.constant(joints_pred, dtype=tf.float32))[1].numpy()

    distance_error = np.linalg.norm(pos_pred - pos_true, axis=1)

    df = pd.DataFrame({
        "x": pos_true[:, 0],
        "y": pos_true[:, 1],
        "z": pos_true[:, 2],
        **{f"q{i+1}_pred": joints_pred[:, i] for i in range(6)},
        "x_pred": pos_pred[:, 0],
        "y_pred": pos_pred[:, 1],
        "z_pred": pos_pred[:, 2],
        "distance_error": distance_error
    })

    df.to_csv(save_path, index=False)
    print(f"Test predictions saved to {save_path}")

def generate_plots(input_path):
    df = pd.read_csv(input_path)

    # ===== 繪製誤差直方圖 =====
    plot_distance_error(df, "results/distance_error_histogram.png", bin_width=0.0025, show=False)

    # ===== 繪製誤差向量 =====
    plot_distance_error_vectors(df, "results/error_vectors.png", show=True)

    # ===== 繪製關節角度分布 =====
    plot_joint_angle_distribution(df, bin_width=0.025, show=False)

    print("[INFO] 完成所有輸出，結果已儲存至 output/")