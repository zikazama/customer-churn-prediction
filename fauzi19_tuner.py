"""Modul Tuner: pencarian hyperparameter otomatis dengan Keras Tuner."""

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.tuner.component import TunerFnResult
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


def _build_model(hp):
    """Model yang sama dengan Trainer, dengan hyperparameter yang dicari Tuner."""
    units = hp.Choice("units", [32, 64, 128])
    dropout_rate = hp.Choice("dropout_rate", [0.1, 0.2, 0.3])
    learning_rate = hp.Choice("learning_rate", [0.01, 0.001, 0.0005])

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


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Entry point Tuner: RandomSearch dengan objective val_auc."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, repeat=True)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, repeat=True)

    tuner = kt.RandomSearch(
        _build_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=6,
        seed=42,
        directory=fn_args.working_dir,
        project_name="churn_tuning",
    )
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 2,
        },
    )
