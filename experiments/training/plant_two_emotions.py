""" This experiment tries training a plant classifier with only
two emotions, calm and excited. """

from typing import Any, Dict

import numpy as np
import tensorflow as tf

from src.data.balanced_plant_exp_reader import (
    BalancedPlantExperimentDataReader,
    Set,
)
from src.utils.metrics import accuracy, per_class_accuracy


def map_emotions(data, labels):
    """
    Convert labels from seven to two emotions.
    Labels refer to:
        0: anger        ->  excited
        1: surprise     ->  excited
        2: disgust      ->  excited
        3: joy          ->  calm (could be both)
        4: fear         ->  excited
        5: sadness      ->  calm
        6: neutral      ->  calm
    """
    new_labels = np.zeros((labels.shape[0], 2))
    conversion_dict = {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 0}
    for old_val, new_val in conversion_dict.items():
        new_labels[:, new_val] += labels[:, old_val]
    new_labels = new_labels.astype(np.float32)
    return data.astype(np.float32), new_labels


def map_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    dataset = dataset.map(
        lambda x, y: tf.numpy_function(
            func=map_emotions,
            inp=[x, y],
            Tout=(tf.float32, tf.float32),
        )
    )
    return dataset


def compute_mfccs(
    audio_tensor: tf.Tensor, parameters: Dict = None
) -> tf.Tensor:
    parameters = parameters or {}
    num_mfcc = parameters.get("num_mfcc", 20)
    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(
        audio_tensor, frame_length=2500, frame_step=1250, fft_length=2500
    )
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 100.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        10000,
        lower_edge_hertz,
        upper_edge_hertz,
    )
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1
    )
    mel_spectrograms.set_shape(
        spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]
        )
    )

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 40.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[
        ..., :num_mfcc
    ]

    return mfccs


def get_labels(dataset: tf.data.Dataset) -> np.ndarray:
    labels = np.empty((0,))
    for batch, blabels in dataset:
        labels = np.concatenate([labels, np.argmax(blabels, axis=1)], axis=0)
    return labels


def create_model(parameters: Dict[str, Any]) -> tf.keras.Model:
    dropout = parameters.get("dropout", 0.2)
    lstm_units = parameters.get("lstm_units", 256)
    input = tf.keras.layers.Input(shape=(400,), dtype=tf.float32, name="raw")
    reshaped = tf.expand_dims(input, 2)
    out = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True)
    )(reshaped)
    out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))(out)
    out = tf.keras.layers.Dropout(dropout)(out)
    out = tf.keras.layers.Dense(1024, activation="relu")(out)
    out = tf.keras.layers.Dropout(dropout)(out)
    out = tf.keras.layers.Dense(2, activation="softmax")(out)
    model = tf.keras.Model(input, out)
    return model


def balance_dataset(
    dataset: tf.data.Dataset, parameters: Dict = None
) -> tf.data.Dataset:
    """
    Main data reading function which reads the plant data into a dataset.
    This function balances the different classes in the dataset.

    :param dataset: Dataset to balance
    :param parameters: Additional parameters
    :return: The tensorflow Dataset instance
    """
    parameters = parameters or {}
    batch_size = parameters.get("batch_size", 64)
    class_data = [np.empty((0, 400)) for _ in range(2)]
    class_datasets = []
    class_names = ["calm", "excited"]
    for plant_data, labels in dataset:
        plant_class = np.argmax(labels.numpy(), axis=1)
        plant_data = plant_data.numpy()
        for index in range(2):
            class_data[index] = np.concatenate(
                [class_data[index], plant_data[plant_class == index, :]],
                axis=0,
            )
    total_count = sum([cd.shape[0] for cd in class_data])
    for index, class_name in enumerate(class_names):
        labels = np.zeros((class_data[index].shape[0], 2))
        labels[:, index] = 1
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor(class_data[index]),
                tf.convert_to_tensor(labels),
            )
        )
        dataset = dataset.repeat()
        class_datasets.append(dataset)

    resampled_ds = tf.data.Dataset.sample_from_datasets(
        class_datasets, weights=[1 / 2.0] * 2
    )
    resampled_ds = resampled_ds.take(total_count).batch(batch_size)
    return resampled_ds


def main():
    reader = BalancedPlantExperimentDataReader()
    train_parameters = {
        "epochs": 1000,
        "patience": 100,
        "batch_size": 64,
        "preprocess": True,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "label_mode": "both",
        "num_mfcc": 60,
        "window": 20,
        "hop": 10,
        "balanced": False,
        "shuffle": False,
    }

    train_data = balance_dataset(
        map_dataset(
            reader.get_seven_emotion_data(Set.TRAIN, 64, train_parameters)
        )
    )
    val_data = balance_dataset(
        map_dataset(
            reader.get_seven_emotion_data(Set.VAL, 64, train_parameters)
        )
    )
    test_data = map_dataset(
        reader.get_seven_emotion_data(Set.TEST, 64, train_parameters)
    )

    model = create_model(parameters=train_parameters)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_parameters.get("patience", 10),
            restore_best_weights=True,
        )
    ]
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=train_parameters.get("learning_rate", 0.001)
    )
    metrics = [tf.metrics.CategoricalAccuracy()]
    loss = tf.keras.losses.CategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    _ = model.fit(
        x=train_data,
        validation_data=val_data,
        epochs=train_parameters.get("epochs", 50),
        callbacks=callbacks,
    )
    predictions = np.argmax(model.predict(test_data), axis=1)
    labels = get_labels(test_data)
    print(f"Accuracy: {accuracy(labels, predictions)}")
    print(f"Per Class Accuracy: {per_class_accuracy(labels, predictions)}")


if __name__ == "__main__":
    main()
