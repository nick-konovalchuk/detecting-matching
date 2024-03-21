from docarray import BaseDoc
from docarray.documents import ImageDoc
from docarray.typing import NdArray


class LabeledImage(ImageDoc):
    label: str


class Detection(BaseDoc):
    parent_id: str
    score: float
    crop: NdArray
    bbox: NdArray


class Match(BaseDoc):
    parent_id: str
    match_url: str
    label: str
    score: float
    bbox: NdArray
