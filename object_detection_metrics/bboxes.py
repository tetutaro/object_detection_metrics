#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Union
from pydantic import BaseModel, validator
import numpy as np


class TrueBBox(BaseModel):
    class_id: int
    bbox: List[Union[int, float]]

    @validator('class_id')
    def check_class_id(cls: TrueBBox, v: int) -> int:
        if v < 0:
            raise ValueError('class_id must be >= 0')
        return v

    @validator('bbox')
    def check_bbox(
        cls: TrueBBox,
        v: List[Union[int, float]]
    ) -> List[float]:
        if len(v) != 4:
            raise ValueError('length of bbox must be 4')
        for vv in v:
            if vv < 0:
                raise ValueError('position must be >= 0')
        if v[0] >= v[2]:
            raise ValueError('min_x is greater than max_x')
        if v[1] >= v[3]:
            raise ValueError('min_y is greater than max_y')
        return [float(x) for x in v]


class PredBBox(TrueBBox):
    class_id: int
    bbox: List[Union[int, float]]
    score: float

    @validator('score')
    def check_score(cls: PredBBox, v: float) -> float:
        if v < 0.0 or 1.0 < v:
            raise ValueError('score must be between 0 and 1')
        return v


class GroundTruth(BaseModel):
    image_id: str
    bboxes: List[TrueBBox]

    def to_ndarray(self: GroundTruth) -> np.ndarray:
        if len(self.bboxes) > 0:
            bboxes = np.stack([
                np.array(bbox.bbox) for bbox in self.bboxes
            ])
        else:
            bboxes = np.empty((0, 4))
        classes = np.array([
            bbox.class_id for bbox in self.bboxes
        ])[:, np.newaxis]
        array = np.concatenate(
            (bboxes, classes), axis=1
        )
        return array


class Prediction(BaseModel):
    image_id: str
    bboxes: List[PredBBox]

    def to_ndarray(self: Prediction) -> np.ndarray:
        if len(self.bboxes) > 0:
            bboxes = np.stack([
                np.array(bbox.bbox) for bbox in self.bboxes
            ])
        else:
            bboxes = np.empty((0, 4))
        classes = np.array([
            bbox.class_id for bbox in self.bboxes
        ])[:, np.newaxis]
        scores = np.array([
            bbox.score for bbox in self.bboxes
        ])[:, np.newaxis]
        array = np.concatenate(
            (bboxes, scores, classes), axis=1
        )
        return array[
            np.argsort(array[:, 4])[::-1]
        ]
