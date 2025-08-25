import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from dataset_utils import load_dataset, split_dataset
from model_utils import build_IK_model, position_loss_distance
from plot_utils import plot_loss
from evaluate_ur10_ik import evaluate_model_metrics, save_test_predictions, generate_plots

if __name__ == "__main__":
    csv_path = "data/positions_dataset.csv"
    X = load_dataset(csv_path)
    X_train, X_val, X_test = split_dataset(X)

    model = build_IK_model(3, 200, 3, 6)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=position_loss_distance # Add "metrics=[position_loss_distance]" if position_loss_distance is not the loss
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=500, batch_size=64,
        callbacks=[early_stop]
    )

    plot_loss(history, "results/loss_curve.png")

    evaluate_model_metrics(model, X_test)
    save_test_predictions(model, X_test, "data/test_predictions.csv")
    generate_plots(input_path="data/test_predictions.csv")

    model.save("models/ik_model.h5")
    print("模型已儲存為 models/ik_model.h5")
