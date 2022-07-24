"""File that aims to visualize the facial images data"""

import numpy as np
import tensorflow as tf
import umap
import umap.plot

from src.data.image_data_reader import ImageDataReader, Set


def embedding_model() -> tf.keras.Model:
    save_path = "models/image/vgg16"
    model = tf.keras.models.load_model(save_path)
    new_model = tf.keras.Model(model.input, model.layers[-3].output)

    return new_model


if __name__ == "__main__":  # pragma: no cover
    dr = ImageDataReader()
    dataset = dr.get_seven_emotion_data(
        Set.TEST, batch_size=1000, parameters={"shuffle": False}
    ).map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))
    model = embedding_model()
    data = np.empty((0, 1000))
    for images, labels in dataset:
        embeddings = model(images).numpy()
        data = np.concatenate([data, embeddings], axis=0)
    labels = dr.get_labels(Set.TEST)
    reducer = umap.UMAP(random_state=42)
    reducer.fit(data)
    umap.plot.points(reducer, labels=labels)
    umap.plot.plt.show()
