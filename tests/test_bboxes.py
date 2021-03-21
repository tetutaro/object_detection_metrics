#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from unittest import TestCase, main
from pydantic import ValidationError
from object_detection_metrics.bboxes import (
    TrueBBox, PredBBox, GroundTruth, Prediction
)


class TestTrueBBox(TestCase):
    def test_true(self: TestTrueBBox) -> None:
        _ = TrueBBox(
            category_id=0,
            bbox=[1, 2, 3, 4]
        )
        _ = TrueBBox(
            category_id=1,
            bbox=[1.1, 2.2, 3.3, 4.4]
        )
        _ = TrueBBox(
            category_id=0,
            bbox=[1, 2, 3, 4],
            hoge='hogehoge'
        )
        return

    def test_false_input_fields(self: TestTrueBBox) -> None:
        with self.assertRaises(ValidationError):
            _ = TrueBBox(
                bbox=[1, 2, 3, 4]
            )
        with self.assertRaises(ValidationError):
            _ = TrueBBox(
                category_id=0,
                hoge='hogehoge'
            )
        return

    def test_false_category_id(self: TestTrueBBox) -> None:
        with self.assertRaises(ValidationError):
            _ = TrueBBox(
                category_id='hoge',
                bbox=[1, 2, 3, 4]
            )
        with self.assertRaises(ValueError):
            _ = TrueBBox(
                category_id=-1,
                bbox=[1, 2, 3, 4]
            )
        return

    def test_false_bbox(self: TestTrueBBox) -> None:
        with self.assertRaises(ValidationError):
            _ = TrueBBox(
                category_id=0,
                bbox=1.0
            )
        with self.assertRaises(ValidationError):
            _ = TrueBBox(
                category_id=0,
                bbox='hoge',
            )
        with self.assertRaises(ValidationError):
            _ = TrueBBox(
                category_id=0,
                bbox=[1, 2, 3, 'hoge']
            )
        with self.assertRaises(ValueError):
            _ = TrueBBox(
                category_id=0,
                bbox=[1, 2, 3, -4]
            )
        with self.assertRaises(ValueError):
            _ = TrueBBox(
                category_id=0,
                bbox=[1, 2, 3],
            )
        with self.assertRaises(ValueError):
            _ = TrueBBox(
                category_id=0,
                bbox=[1, 2, 3, 4, 5]
            )
        with self.assertRaises(ValueError):
            _ = TrueBBox(
                category_id=0,
                bbox=[3, 2, 1, 4]
            )
        with self.assertRaises(ValueError):
            _ = TrueBBox(
                category_id=0,
                bbox=[1, 4, 3, 2]
            )
        with self.assertRaises(ValueError):
            _ = TrueBBox(
                category_id=0,
                bbox=[3, 4, 1, 2]
            )
        return


class TestPredBBox(TestCase):
    def test_true(self: TestPredBBox) -> None:
        _ = PredBBox(
            category_id=0,
            bbox=[1, 2, 3, 4],
            score=0.25
        )
        _ = PredBBox(
            category_id=1,
            bbox=[1.1, 2.2, 3.3, 4.4],
            score=0.75
        )
        _ = PredBBox(
            category_id=0,
            bbox=[1, 2, 3, 4],
            score=0.25,
            hoge='hogehoge'
        )
        return

    def test_false_input_fields(self: TestPredBBox) -> None:
        with self.assertRaises(ValidationError):
            _ = PredBBox(
                bbox=[1, 2, 3, 4],
                score=0.25
            )
        with self.assertRaises(ValidationError):
            _ = PredBBox(
                category_id=0,
                score=0.25
            )
        with self.assertRaises(ValidationError):
            _ = PredBBox(
                category_id=0,
                bbox=[1, 2, 3, 4],
                hoge='hogehoge'
            )
        return

    def test_false_category_id(self: TestPredBBox) -> None:
        with self.assertRaises(ValidationError):
            _ = PredBBox(
                category_id='hoge',
                bbox=[1, 2, 3, 4],
                score=0.25,
            )
        with self.assertRaises(ValueError):
            _ = PredBBox(
                category_id=-1,
                bbox=[1, 2, 3, 4],
                score=0.25
            )
        return

    def test_false_bbox(self: TestPredBBox) -> None:
        with self.assertRaises(ValidationError):
            _ = PredBBox(
                category_id=0,
                bbox=1.0,
                score=0.25
            )
        with self.assertRaises(ValidationError):
            _ = PredBBox(
                category_id=0,
                bbox='hoge',
                score=0.25
            )
        with self.assertRaises(ValidationError):
            _ = PredBBox(
                category_id=0,
                bbox=[1, 2, 3, 'hoge'],
                score=0.25
            )
        with self.assertRaises(ValueError):
            _ = PredBBox(
                category_id=0,
                bbox=[1, 2, 3, -4],
                score=0.25
            )
        with self.assertRaises(ValueError):
            _ = PredBBox(
                category_id=0,
                bbox=[1, 2, 3],
                score=0.25
            )
        with self.assertRaises(ValueError):
            _ = PredBBox(
                category_id=0,
                bbox=[1, 2, 3, 4, 5],
                score=0.25
            )
        with self.assertRaises(ValueError):
            _ = PredBBox(
                category_id=0,
                bbox=[3, 2, 1, 4],
                score=0.25
            )
        with self.assertRaises(ValueError):
            _ = PredBBox(
                category_id=0,
                bbox=[1, 4, 3, 2],
                score=0.25
            )
        with self.assertRaises(ValueError):
            _ = PredBBox(
                category_id=0,
                bbox=[3, 4, 1, 2],
                score=0.25
            )
        return

    def test_false_score(self: TestPredBBox) -> None:
        with self.assertRaises(ValidationError):
            _ = PredBBox(
                category_id=0,
                bbox=[1, 2, 3, 4],
                score='hoge'
            )
        with self.assertRaises(ValidationError):
            _ = PredBBox(
                category_id=0,
                bbox=[1, 2, 3, 4],
                score=-0.5
            )
        with self.assertRaises(ValidationError):
            _ = PredBBox(
                category_id=0,
                bbox=[1, 2, 3, 4],
                score=1.5
            )
        return


class TestGroundTruth(TestCase):
    bboxes = [
        {
            'category_id': 0,
            'bbox': [1, 2, 3, 4],
        },
        {
            'category_id': 1,
            'bbox': [1.1, 2.2, 3.3, 4.4],
        },
    ]
    trues = [
        {
            'image_id': '1234',
            'bboxes': [
                {
                    'category_id': 0,
                    'bbox': [1, 2, 3, 4],
                },
                {
                    'category_id': 1,
                    'bbox': [1.1, 2.2, 3.3, 4.4],
                },
            ],
        },
        {
            'image_id': 5678,
            'bboxes': [],
        },
    ]

    def test_true(self: TestGroundTruth) -> None:
        _ = GroundTruth(
            image_id='1234',
            bboxes=self.bboxes
        )
        _ = GroundTruth(
            image_id='5678',
            bboxes=[]
        )
        _ = GroundTruth(
            image_id=1234,
            bboxes=self.bboxes,
            hoge='hogehoge'
        )
        for true in self.trues:
            _ = GroundTruth(**true)
        return

    def test_false_input_fields(self: TestGroundTruth) -> None:
        with self.assertRaises(ValidationError):
            _ = GroundTruth(
                bboxes=self.bboxes
            )
        with self.assertRaises(ValidationError):
            _ = GroundTruth(
                image_id='1234',
                hoge='hogehoge'
            )
        return

    def test_false_image_id(self: TestGroundTruth) -> None:
        with self.assertRaises(ValidationError):
            _ = GroundTruth(
                image_id=[1234],
                bboxes=self.bboxes
            )


class TestPrediction(TestCase):
    bboxes = [
        {
            'category_id': 0,
            'bbox': [1, 2, 3, 4],
            'score': 0.25,
        },
        {
            'category_id': 1,
            'bbox': [1.1, 2.2, 3.3, 4.4],
            'score': 0.75,
        },
    ]
    preds = [
        {
            'image_id': '1234',
            'bboxes': [
                {
                    'category_id': 0,
                    'bbox': [1, 2, 3, 4],
                    'score': 0.25,
                },
                {
                    'category_id': 1,
                    'bbox': [1.1, 2.2, 3.3, 4.4],
                    'score': 0.75,
                },
            ],
        },
        {
            'image_id': 5678,
            'bboxes': [],
        },
    ]

    def test_true(self: TestPrediction) -> None:
        _ = Prediction(
            image_id='1234',
            bboxes=self.bboxes
        )
        _ = Prediction(
            image_id='5678',
            bboxes=[]
        )
        _ = Prediction(
            image_id=1234,
            bboxes=self.bboxes,
            hoge='hogehoge'
        )
        for pred in self.preds:
            _ = Prediction(**pred)
        return

    def test_false_input_fields(self: TestPrediction) -> None:
        with self.assertRaises(ValidationError):
            _ = Prediction(
                bboxes=self.bboxes
            )
        with self.assertRaises(ValidationError):
            _ = Prediction(
                image_id='1234',
                hoge='hogehoge'
            )
        return

    def test_false_image_id(self: TestPrediction) -> None:
        with self.assertRaises(ValidationError):
            _ = Prediction(
                image_id=[1234],
                bboxes=self.bboxes,
            )
        return


if __name__ == '__main__':
    main()
