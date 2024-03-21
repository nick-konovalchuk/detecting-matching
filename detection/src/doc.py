from typing import List

from docarray import BaseDoc
from docarray.typing import ImageUrl, NdArray
from pydantic import Field


class ImageUrlDoc(BaseDoc):
    url: ImageUrl


class DetectionResult(BaseDoc):
    url: ImageUrl
    scores: NdArray
    detections: NdArray


class Detection(BaseDoc):
    parent_id: str
    score: float
    crop: NdArray
    bbox: NdArray

