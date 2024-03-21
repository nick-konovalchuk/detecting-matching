import cv2
from docarray import BaseDoc
from docarray import DocList
from docarray.typing import ImageUrl
from docarray.typing import NdArray
from jina import Executor
from jina import requests
import numpy as np
import tensorflow as tf


class DetectionResult(BaseDoc):
    id: int
    url: ImageUrl
    scores: NdArray
    detections: NdArray


class ClassificationResult(BaseDoc):
    id: int
    class_: int


class ClassificationExecutor(Executor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = tf.keras.applications.EfficientNetB0()

    @requests
    def classify(self, docs: DocList[DetectionResult], **kwargs) -> DocList[ClassificationResult]:
        ids = []
        batch = []
        for doc in docs:
            tensor = doc.url.load()
            for det in doc.detections:
                ids.append(doc.id)
                crop = tensor[det[1] : det[3], det[0] : det[2]]
                batch.append(cv2.resize(crop, (224, 224)))

        preds = tf.argmax(self.model(np.array(batch)), -1).numpy()

        return DocList(
            [ClassificationResult(id=id_, class_=class_) for id_, class_ in zip(ids, preds)]
        )
