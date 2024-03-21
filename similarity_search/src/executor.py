from typing import List

from docarray import DocList
from docarray.array.doc_vec.doc_vec import TensorFlowTensor
from docarray.index import InMemoryExactNNIndex
from docarray.typing import NdArray
from jina import Executor
from jina import requests
import tensorflow as tf

from .doc import Detection
from .doc import LabeledImage
from .doc import Match


REF_IDS = list(range(6))
REF_LABELS = ["cat"] * 3 + ["dog"] * 3
REF_URLS = [
    "https://drive.google.com/uc?export=download&id=1wM8BcjExpkzdQTfDvCQXz0kcxtGRLbpq",
    "https://drive.google.com/uc?export=download&id=1b7UXIsV-9QDGRYf-DwPQF6ERrEfkK1AQ",
    "https://drive.google.com/uc?export=download&id=1aY67nHgOuajgbMVRNmSqwQHZI5nJYpSR",
    "https://drive.google.com/uc?export=download&id=1jp1mbqK44dSaGqQukrSEHYdCJ1X-ZAK9",
    "https://drive.google.com/uc?export=download&id=1Rq4dTLwQU-OPBdMd3NCnSPdLDv6mpxzv",
    "https://drive.google.com/uc?export=download&id=12OedKd4IqbYq-3Q5oxaO545YeTMI7wfs",
]


class SimilaritySearchExecutor(Executor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = tf.keras.applications.EfficientNetB0(include_top=False)

        refs = DocList[LabeledImage](
            [
                LabeledImage(id=id_, label=label, url=url)
                for id_, label, url in zip(REF_IDS, REF_LABELS, REF_URLS)
            ]
        )

        refs.embedding = self.embed([i.load() for i in refs.url])

        self.index = InMemoryExactNNIndex[LabeledImage](refs)

    def embed(self, images: List[NdArray]) -> tf.Tensor:
        batch = tf.stack([tf.image.resize(image, (224, 224)) for image in images])
        return tf.reduce_mean(self.model(batch), (1, 2))

    @requests
    def search(self, docs: DocList[Detection], **kwargs) -> DocList[Match]:
        queries = TensorFlowTensor(self.embed(docs.crop))
        neighbour_groups, score_groups = self.index.find_batched(
            queries, limit=1, search_field="embedding"
        )

        return DocList[Match](
            [
                Match(
                    id=doc.id,
                    bbox=doc.bbox,
                    parent_id=doc.parent_id,
                    match_url=neighbours[0].url,
                    label=neighbours[0].label,
                    score=scores.tensor.numpy().item(),
                )
                for doc, neighbours, scores in zip(docs, neighbour_groups, score_groups)
            ]
        )
