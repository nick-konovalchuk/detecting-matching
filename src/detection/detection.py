from typing import Dict
from typing import List

from docarray import BaseDoc
from docarray import DocList
from docarray.typing import ImageUrl
from docarray.typing import NdArray
from jina import Executor
from jina import requests
import numpy as np
import torch
from transformers import pipeline


class Image(BaseDoc):
    id: int
    url: ImageUrl


class DetectionResult(BaseDoc):
    id: int
    url: ImageUrl
    scores: NdArray
    detections: NdArray


def reformat_detections(images: DocList, detections: List[Dict]) -> List[DetectionResult]:
    return [
        DetectionResult(
            id=im.id,
            url=im.url,
            scores=np.array([det["score"] for det in dets]),
            detections=np.array([list(det["box"].values()) for det in dets]),
        )
        for im, dets in zip(images, detections)
    ]


class DetectionExecutor(Executor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pipeline = pipeline("object-detection", model="facebook/detr-resnet-50")

    @requests
    def detect(self, docs: DocList[Image], **kwargs) -> DocList[DetectionResult]:
        with torch.inference_mode():
            pred = self.pipeline(docs.url)

        return DocList(reformat_detections(docs, pred))
