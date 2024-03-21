import numpy as np
import torch
from PIL import Image
from docarray import DocList
from jina import Executor, requests
from transformers import pipeline
from requests import get

from .doc import ImageUrlDoc, Detection


class DetectionExecutor(Executor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pipeline = pipeline("object-detection", model="facebook/detr-resnet-50")

    @requests
    def detect(self, docs: DocList[ImageUrlDoc], **kwargs) -> DocList[Detection]:
        with torch.inference_mode():
            preds = self.pipeline(docs.url)

        detections = []
        for doc, pred in zip(docs, preds):
            image = Image.open(get(doc.url, stream=True).raw)
            for det in pred:
                box = det["box"]
                crop = image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
                detections.append(
                    Detection(
                        parent_id=doc.id,
                        score=det["score"],
                        crop=np.asarray(crop)
                    )
                )

        return DocList(detections)
