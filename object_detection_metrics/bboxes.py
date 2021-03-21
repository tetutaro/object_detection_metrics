#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Union
from pydantic import BaseModel, validator
import numpy as np


class TrueBBox(BaseModel):
    '''ground truth bounding box

    Attributes:
        category_id (str): id of detected object
        bbox (List): bouding box (min_x, min_y, max_x, max_y)
    '''
    category_id: int
    bbox: List[Union[int, float]]

    @validator('category_id')
    def check_category_id(cls: TrueBBox, v: int) -> int:
        if v < 0:
            raise ValueError('category_id must be >= 0')
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
        if v[0] > v[2]:
            raise ValueError(f'minx({v[0]}) is greater than maxx({v[2]})')
        if v[1] > v[3]:
            raise ValueError(f'miny({v[1]}) is greater than maxy({v[3]})')
        return [float(x) for x in v]


class PredBBox(TrueBBox):
    '''predicted bounding box

    Attributes:
        category_id (str): id of detected object
        bbox (List): bouding box (min_x, min_y, max_x, max_y)
        score: confidence score
    '''
    category_id: int
    bbox: List[Union[int, float]]
    score: float

    @validator('score')
    def check_score(cls: PredBBox, v: float) -> float:
        if v < 0.0 or 1.0 < v:
            raise ValueError('score must be between 0 and 1')
        return v


class GroundTruth(BaseModel):
    '''ground truth bounding boxes of an image

    Attributes:
        image_id (str): image id
        bboxes (List[TrueBBox]): ground truth bouding boxes
    '''
    image_id: str
    bboxes: List[TrueBBox]

    def to_ndarray(self: GroundTruth) -> np.ndarray:
        '''convert Prediction to np.ndarray

        * column 0-3: bounding boxes (min_x, min_y, max_x, max_y)
        * column 4: category ids

        Returns:
            np.ndarray: N x 6 matrix
        '''
        # bounding boxes
        # bounding boxes
        if len(self.bboxes) > 0:
            bboxes = np.stack([
                np.array(bbox.bbox) for bbox in self.bboxes
            ])
        else:
            bboxes = np.empty((0, 4))
        # category ids
        categories = np.array([
            bbox.category_id for bbox in self.bboxes
        ])[:, np.newaxis]
        # unite bounding boxes and category ids
        array = np.concatenate(
            (bboxes, categories), axis=1
        )
        return array


class Prediction(BaseModel):
    '''predicted bounding boxes of an image

    Attributes:
        image_id (str): image id
        bboxes (List[TrueBBox]): predicted bouding boxes
    '''
    image_id: str
    image_id: str
    bboxes: List[PredBBox]

    def to_ndarray(self: Prediction) -> np.ndarray:
        '''convert Prediction to np.ndarray

        * column 0-3: bounding boxes (min_x, min_y, max_x, max_y)
        * column 4: confidence scores
        * column 5: category ids
        * predicted bouding boxes are sorted
          in descending order of confidence score

        Returns:
            np.ndarray: N x 6 matrix
        '''
        # bounding boxes
        if len(self.bboxes) > 0:
            bboxes = np.stack([
                np.array(bbox.bbox) for bbox in self.bboxes
            ])
        else:
            bboxes = np.empty((0, 4))
        # category ids
        categories = np.array([
            bbox.category_id for bbox in self.bboxes
        ])[:, np.newaxis]
        # confidence scores
        scores = np.array([
            bbox.score for bbox in self.bboxes
        ])[:, np.newaxis]
        # unite bounding boxes, confidence scores and cagetory ids
        array = np.concatenate(
            (bboxes, scores, categories), axis=1
        )
        # sort in descending order of confidence score
        return array[
            np.argsort(array[:, 4])[::-1]
        ]
