"""Modul Trainer: model Keras untuk prediksi churn, memakai hyperparameter hasil Tuner."""

import keras
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "Churn"
LABEL_TRANSFORMED = f"{LABEL_KEY}_xf"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_TRANSFORMED = [f"{key}_xf" for key in NUMERIC_FEATURES]
CATEGORICAL_TRANSFORMED = [f"{key}_xf" for key in CATEGORICAL_FEATURES]
VOCAB_SIZE = 100
EMBEDDING_DIM = 8

# Hyperparameter bawaan bila Trainer dijalankan tanpa hasil Tuner.
DEFAULT_HYPERPARAMETERS = {"units": 64, "dropout_rate": 0.2, "learning_rate": 0.001}


def _input_fn(file_pattern, tf_transform_output, batch_size=64, repeat=True):
    """Membaca TFRecord hasil Transform menjadi dataset (fitur, label)."""
    feature_spec = tf_transform_output.transformed_feature_spec()
    if isinstance(file_pattern, str):
        file_pattern = [file_pattern]
    files = []
    for pattern in file_pattern:
        matched = tf.io.gfile.glob(pattern)
        files.extend(matched if matched else [pattern])
    dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")

    def _parse(serialized):
        parsed = tf.io.parse_example(serialized, feature_spec)
        label = parsed.pop(LABEL_TRANSFORMED)
        for key in parsed:
            parsed[key] = tf.expand_dims(parsed[key], -1)
        label = tf.expand_dims(tf.cast(label, tf.float32), -1)
        return parsed, label

    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat()
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _hyperparameter(hparams, name):
    """Mengambil nilai hyperparameter dari hasil Tuner atau nilai bawaan."""
    if hparams is not None and name in hparams.values:
        return hparams.get(name)
    return DEFAULT_HYPERPARAMETERS[name]


def build_model(hparams=None):
    """Membangun model Keras dua cabang (numerik dan kategorikal)."""
    units = int(_hyperparameter(hparams, "units"))
    dropout_rate = float(_hyperparameter(hparams, "dropout_rate"))
    learning_rate = float(_hyperparameter(hparams, "learning_rate"))

    numeric_inputs = [
        tf.keras.Input(shape=(1,), name=key, dtype=tf.float32) for key in NUMERIC_TRANSFORMED
    ]
    categorical_inputs = [
        tf.keras.Input(shape=(1,), name=key, dtype=tf.int64) for key in CATEGORICAL_TRANSFORMED
    ]

    numeric = tf.keras.layers.Concatenate()(numeric_inputs)
    numeric = tf.keras.layers.Dense(units, activation="relu")(numeric)
    numeric = tf.keras.layers.Dropout(dropout_rate)(numeric)
    numeric = tf.keras.layers.Dense(max(units // 2, 8), activation="relu")(numeric)

    embeddings = []
    for inp in categorical_inputs:
        emb = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM)(inp)
        embeddings.append(tf.keras.layers.Flatten()(emb))
    categorical = tf.keras.layers.Concatenate()(embeddings)
    categorical = tf.keras.layers.Dense(max(units // 2, 8), activation="relu")(categorical)

    x = tf.keras.layers.Concatenate()([numeric, categorical])
    x = tf.keras.layers.Dense(max(units // 2, 8), activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="churn_probability")(x)

    model = tf.keras.Model(inputs=numeric_inputs + categorical_inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _export_for_serving(model, tf_transform_output, export_dir):
    """Mengekspor SavedModel yang kompatibel dengan TensorFlow Serving (Keras 3 ExportArchive).

    Dua signature dibuat:
    - serving_default: menerima tf.Example serial, menerapkan Transform, lalu memprediksi.
    - transform_features: dipakai Evaluator untuk menyiapkan fitur dan label.
    """
    tft_layer = tf_transform_output.transform_features_layer()
    input_signature = [tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]

    @tf.function(input_signature=input_signature)
    def serve_tf_examples_fn(serialized_tf_examples):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(LABEL_KEY)
        raw_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        transformed_features = tft_layer(raw_features)
        return {"outputs": model(transformed_features)}

    @tf.function(input_signature=input_signature)
    def transform_features_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        return tft_layer(raw_features)

    archive = keras.export.ExportArchive()
    archive.track(model)
    archive.track(tft_layer)
    archive.add_endpoint("serving_default", serve_tf_examples_fn)
    archive.add_endpoint("transform_features", transform_features_fn)
    archive.write_out(export_dir)


def run_fn(fn_args: FnArgs):
    """Entry point Trainer."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, repeat=True)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, repeat=True)

    hparams = None
    if fn_args.hyperparameters:
        hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)
        print("Hyperparameter dari Tuner:", hparams.values)
    else:
        print("Tuner tidak digunakan, memakai hyperparameter bawaan:", DEFAULT_HYPERPARAMETERS)

    model = build_model(hparams)
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=3,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq="batch"),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=2, restore_best_weights=True
            ),
        ],
    )

    _export_for_serving(model, tf_transform_output, fn_args.serving_model_dir)
