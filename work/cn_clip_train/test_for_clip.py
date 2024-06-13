# first clip embedding
from dataclasses import dataclass


@dataclass
class ImageInfo:
    url: str
    embedding: list
    tag_id: int
    car_id: int
