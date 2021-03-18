#!/usr/bin/env python
# -*- coding:utf-8 -*-
from unittest import TestSuite, makeSuite
from .test_bboxes import (
    TestTrueBBox, TestPredBBox,
    TestGroundTruth, TestPrediction
)


def suite() -> TestSuite:
    suite = TestSuite()
    suite.addTests(makeSuite(TestTrueBBox))
    suite.addTests(makeSuite(TestPredBBox))
    suite.addTests(makeSuite(TestGroundTruth))
    suite.addTests(makeSuite(TestPrediction))
    return suite
