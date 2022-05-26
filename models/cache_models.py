"""This file downloads all tensorflow hub models into a cache folder"""

import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # noqa: F401
from torchvision import models
from transformers import HubertModel, Wav2Vec2Model, Wav2Vec2Processor

if __name__ == "__main__":
    # This script is downloading all the models to a local folder.
    # Running this is not required in general, only when using PCs without
    # internet access.
    assert os.environ["TFHUB_CACHE_DIR"].endswith("models/cache")
    bert_names = [
        "bert_en_uncased_L-2_H-128_A-2",
        "bert_en_uncased_L-4_H-128_A-2",
        "bert_en_uncased_L-4_H-256_A-4",
        "bert_en_uncased_L-2_H-256_A-4",
        "bert_en_uncased_L-6_H-256_A-4",
        "bert_en_uncased_L-4_H-512_A-8",
    ]
    model_names = [
        f"https://tfhub.dev/tensorflow/small_bert/{model_name}/2"
        for model_name in bert_names
    ]
    model_names.append(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    )
    model_names.append(
        "https://tfhub.dev/jeongukjae/distilbert_en_uncased_L-6_H-768_A-12/1"
    )
    model_names.append(
        "https://tfhub.dev/jeongukjae/distilbert_en_uncased_preprocess/2"
    )
    input = tf.keras.layers.Input(
        shape=(48, 48, 3), dtype=tf.float32, name="image"
    )
    model2 = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_tensor=input,
        input_shape=(48, 48, 3),
    )
    model3 = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=input,
        input_shape=(48, 48, 3),
    )

    for model in model_names:
        _ = hub.KerasLayer(model)

    resnet = models.resnet18(pretrained=True)

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/hubert-large-ls960-ft"
    )
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

    processor2 = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    model2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
