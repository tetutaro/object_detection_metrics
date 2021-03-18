#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Dict
import os
from collections import defaultdict
import simplejson as json
from pydantic import BaseModel
from .bboxes import GroundTruth, Prediction
from .calc import BaseEval
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
        self._load(path_true=trues, path_pred=preds)
        return

    @staticmethod
    def _load_jsonl(path: str, model: BaseModel, uniqs: set) -> Dict:
        entries = dict()
        ln = 0
        with open(path, 'rt') as rf:
            raw = rf.readline()
            while raw:
                ln += 1
                try:
                    entry = model(**json.loads(raw))
                except Exception as e:
                    msg = f'{path}:{ln} format broken. ignore.\n{e}'
                    logger.warning(msg)
                else:
                    for bbox in entry.bboxes:
                        uniqs.add(bbox.class_id)
                    entries[entry.image_id] = entry
                finally:
                    raw = rf.readline()
        return entries

    def _load(self: Evaluator, path_true: str, path_pred: str) -> None:
        # load json lines
        unique_classes = set()
        trues = self._load_jsonl(
            path=path_true, model=GroundTruth, uniqs=unique_classes
        )
        preds = self._load_jsonl(
            path=path_pred, model=Prediction, uniqs=unique_classes
        )
        unique_classes = sorted(list(unique_classes))
        # check image_ids are unique in each data
        assert len(list(trues.keys())) == len(set(trues.keys()))
        assert len(list(preds.keys())) == len(set(preds.keys()))
        # get GroundTruth and Prediction which has the same image_id
        self.bases = defaultdict(list)
        for image_id, pred in preds.items():
            true = trues.get(image_id)
            if true is None:
                continue
            true_bboxes = true.to_ndarray()
            pred_bboxes = pred.to_ndarray()
            # divide bounding boxes by class_id
            for class_id in unique_classes:
                true_index = true_bboxes[:, 4] == class_id
                true_count = true_index.astype(int).sum()
                pred_index = pred_bboxes[:, 5] == class_id
                pred_count = pred_index.astype(int).sum()
                if true_count == 0 and pred_count == 0:
                    continue
                # add BaseEval to the bases by class_id
                self.bases[class_id].append(
                    BaseEval(
                        image_id=image_id,
                        class_id=class_id,
                        true=true_bboxes[true_index][:, :-1],
                        pred=pred_bboxes[pred_index][:, :-1]
                    )
                )
        return

    def calc(self: Evaluator) -> None:
        for class_id, bases in sorted(self.bases.items()):
            for base in bases:
                print(base)
        return


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
    evaluator.calc()
    return
