import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ====== UR10 Forward Kinematics ======
# FK in TensorFlow to stay differentiable for gradient-based IK training
def forward_kinematics(joint_angles):
    # DH parameters
    d = tf.constant([0.1273, 0, 0, 0.163941, 0.1157, 0.0922], dtype=tf.float32)
    a = tf.constant([0, -0.612, -0.5723, 0, 0, 0], dtype=tf.float32)
    alpha = tf.constant([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0], dtype=tf.float32)

    batch_size = tf.shape(joint_angles)[0]
    T = tf.eye(4, batch_shape=[batch_size])

    for i in range(6):
        theta = joint_angles[:, i]
        ct = tf.cos(theta)
        st = tf.sin(theta)
        ca = tf.cos(alpha[i])
        sa = tf.sin(alpha[i])

        zeros = tf.zeros_like(theta)
        ones = tf.ones_like(theta)

        Ti = tf.stack([
            tf.stack([   ct,  -st*ca,   st*sa,   a[i]*ct], axis=1),
            tf.stack([   st,   ct*ca,  -ct*sa,   a[i]*st], axis=1),
            tf.stack([zeros, sa*ones, ca*ones, d[i]*ones], axis=1),
            tf.stack([zeros,   zeros,   zeros,      ones], axis=1)
        ], axis=1)

        T = tf.matmul(T, Ti)

    R = T[:, 0:3, 0:3] # Rotation matrix
    pos = T[:, 0:3, 3] # Translation vector
    return R, pos

# ===== position loss =====
# pos_true --ik_model--> joints_pred --forward_kinematics--> pos_pred
def position_loss_mse(pos_true, joints_pred):
    _, pos_pred = forward_kinematics(joints_pred)
    diff = pos_pred - pos_true
    mse = tf.reduce_sum(tf.square(diff), axis=1)
    return tf.reduce_mean(mse)

# pos_true --ik_model--> joints_pred --forward_kinematics--> pos_pred
def position_loss_mae(pos_true, joints_pred):
    _, pos_pred = forward_kinematics(joints_pred)
    diff = pos_pred - pos_true
    mae = tf.reduce_sum(tf.abs(diff), axis=1)
    return tf.reduce_mean(mae)

# pos_true --ik_model--> joints_pred --forward_kinematics--> pos_pred
def position_loss_distance(pos_true, joints_pred):
    _, pos_pred = forward_kinematics(joints_pred)
    diff = pos_pred - pos_true
    dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=1))
    return tf.reduce_mean(dist)

# ====== 建立模型 ======
# This is a multi-layer perceptron (MLP) model
# You can try adding an "orientation" parameter to the input
def build_IK_model(num_hidden_layers, hidden_size, input_dim=3, output_dim=6):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for _ in range(num_hidden_layers):
        model.add(layers.Dense(hidden_size, activation="relu"))

    model.add(layers.Dense(output_dim, activation=None)) # Try more activation (ex: "tanh")
    return model
