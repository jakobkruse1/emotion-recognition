"""File that aims to visualize the facial images data"""

import numpy as np
import tensorflow as tf
import umap
import umap.plot

from src.data.image_data_reader import ImageDataReader, Set


def embedding_model() -> tf.keras.Model:
    input = tf.keras.layers.Input(
        shape=(48, 48, 3), dtype=tf.float32, name="image"
    )
    input = tf.keras.applications.efficientnet.preprocess_input(input)

    model = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_tensor=input,
        input_shape=(48, 48, 3),
    )
    output = model(input)
    output = tf.keras.layers.Flatten()(output)

    return tf.keras.Model(input, output)


if __name__ == "__main__":  # pragma: no cover
    dr = ImageDataReader()
    dataset = dr.get_seven_emotion_data(
        Set.TEST, batch_size=1000, shuffle=False
    ).map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))
    model = embedding_model()
    data = np.empty((0, 5632))
    for images, labels in dataset:
        embeddings = model(images).numpy()
        data = np.concatenate([data, embeddings], axis=0)
    labels = dr.get_labels(Set.TEST)
    reducer = umap.UMAP(random_state=42)
    reducer.fit(data)
    umap.plot.points(reducer, labels=labels)
    umap.plot.plt.show()
