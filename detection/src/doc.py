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
    parent_id: str = Field(default="https://www.purina.co.uk/sites/default/files/2023-03/Hero%20Pedigree%20Cats.jpg")
    score: float
    crop: NdArray
