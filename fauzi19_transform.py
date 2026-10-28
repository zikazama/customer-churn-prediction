"""Modul Transform: preprocessing_fn untuk dataset Telco Customer Churn."""

import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "Churn"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def _flatten(x):
    """Mengubah SparseTensor atau tensor 2D menjadi tensor 1D."""
    if isinstance(x, tf.SparseTensor):
        x = tf.sparse.to_dense(x)
    return tf.reshape(x, [-1])


def preprocessing_fn(inputs):
    """Standardisasi fitur numerik, vocabulary fitur kategorikal, dan label biner."""
    inputs = {key: _flatten(value) for key, value in inputs.items()}
    outputs = {}

    for key in NUMERIC_FEATURES:
        outputs[f"{key}_xf"] = tft.scale_to_z_score(tf.cast(inputs[key], tf.float32))

    for key in CATEGORICAL_FEATURES:
        outputs[f"{key}_xf"] = tft.compute_and_apply_vocabulary(
            inputs[key], num_oov_buckets=1, vocab_filename=key
        )

    outputs[f"{LABEL_KEY}_xf"] = tf.cast(
        tf.equal(inputs[LABEL_KEY], tf.constant("Yes", dtype=tf.string)), tf.int64
    )
    return outputs
