from typing import Dict, List

import numpy as np
from docarray import DocList

from .doc import DetectionResult


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
