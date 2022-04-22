"""This file downloads all tensorflow hub models into a cache folder"""

import os

import tensorflow_hub as hub
import tensorflow_text as text  # noqa: F401

if __name__ == "__main__":
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

    for model in model_names:
        _ = hub.KerasLayer(model)
