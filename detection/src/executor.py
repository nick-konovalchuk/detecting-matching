import numpy as np
import torch
from PIL import Image
from docarray import DocList
from jina import Executor, requests
from requests import get

from .doc import ImageUrlDoc, Detection


from transformers import pipeline


class DetectionExecutor(Executor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pipeline = pipeline(
            task="object-detection",
            model="facebook/detr-resnet-50",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    @requests
    def detect(self, docs: DocList[ImageUrlDoc], **kwargs) -> DocList[Detection]:
        with torch.inference_mode():
            preds = self.pipeline(docs.url)

        detections = []
        for doc, pred in zip(docs, preds):
            image = Image.open(get(doc.url, stream=True).raw)
            for det in pred:
                box = det["box"]
                box = (box['xmin'], box['ymin'], box['xmax'], box['ymax'])
                crop = image.crop(box)
                detections.append(
                    Detection(
                        parent_id=doc.id,
                        score=det["score"],
                        crop=np.asarray(crop),
                        bbox=np.asarray(box)
                    )
                )

        return DocList[Detection](detections)
