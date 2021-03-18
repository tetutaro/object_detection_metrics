#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
import os
import simplejson as json
from .bboxes import GroundTruth, Prediction
import argparse
from logging import getLogger, StreamHandler, Formatter, INFO

logger = getLogger(__file__)
logger.setLevel(INFO)
formatter = Formatter('%(asctime)s: %(levelname)s: %(message)s')
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


class Evaluator(object):
    def __init__(self: Evaluator, trues: str, preds: str) -> None:
        if not os.path.isfile(trues):
            msg = f'trues({trues}) not found'
            raise ValueError(msg)
        if not trues.endswith('.jsonl'):
            msg = f'trues({trues}) must be json line format'
            raise ValueError(msg)
        if not os.path.isfile(preds):
            msg = f'preds({preds}) not found'
            raise ValueError(msg)
        if not preds.endswith('.jsonl'):
            msg = f'preds({preds}) must be json line format'
            raise ValueError(msg)
        self.trues = list()
        ln = 0
        with open(trues, 'rt') as rf:
            raw = rf.readline()
            while raw:
                ln += 1
                try:
                    true = GroundTruth(**json.loads(raw))
                except Exception as e:
                    msg = f'{trues}:{ln} format broken({e}). ignore.'
                    logger.warning(msg)
                else:
                    self.tures.append(true)
                finally:
                    raw = rf.readline()
        self.preds = list()
        ln = 0
        with open(preds, 'rt') as rf:
            raw = rf.readline()
            while raw:
                ln += 1
                try:
                    pred = Prediction(**json.loads(raw))
                except Exception as e:
                    msg = f'{preds}:{ln} format broken({e}). ignore.'
                    logger.warning(msg)
                else:
                    self.preds.append(pred)
                finally:
                    raw = rf.readline()
        return

    def calc() -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description='calculate metrics used to evaluate object detection'
    )
    parser.add_argument(
        '--trues', '-t', required=True, type=str,
        help='the file of ground truth bounding boxes'
    )
    parser.add_argument(
        '--preds', '-p', required=True, type=str,
        help='the file of predicted bounding boxes'
    )
    args = parser.parse_args()
    evaluator = Evaluator(**vars(args))
    evaluator.cals()
    return


if __name__ == '__main__':
    main()
