#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Dict
import os
import time
from collections import defaultdict
import simplejson as json
import numpy as np
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
    '''main class of evaluating object detection algorithm

    Attributes:
        verbose (bool): print verbosely
        unique_classes (List):
            unique class ids of detected objects.
            it may have unused class ids because
            object detection may not be made on
            all images of ground truths.
        aps (Dict[int, float]):
            Average Precisions per IoU threshold.
            the values of key are int(threshold * 100).
            value 100 of key means IoU=0.50:0.95:0.05
        load_start (float): time to start loading
        load_end (float): time to finish loading
        eval_start (float): time to start evaluating
        eval_end (float): time to finish evaluating
        accm_start (float): time to start accumulating
        accm_end (float): time to finish accumulating
    '''

    def __init__(
        self: Evaluator,
        trues: str,
        preds: str,
        verbose: bool = False
    ) -> None:
        '''initialize function of Evaluator

        * check paths of given json lines format files
        * initialize attributes
        * call self._load()

        Args:
            trues (str): the path of ground truth json lines format file
            preds (str): the path of prediction json lines format file
            verbose (bool): print verbosely or not
        '''
        # check paths
        if not os.path.isfile(trues):
            msg = f'trues({trues}) not found'
            raise ValueError(msg)
        if not trues.endswith('.jsonl'):
            msg = f'trues({trues}) must be json lines format'
            raise ValueError(msg)
        if not os.path.isfile(preds):
            msg = f'preds({preds}) not found'
            raise ValueError(msg)
        if not preds.endswith('.jsonl'):
            msg = f'preds({preds}) must be json lines format'
            raise ValueError(msg)
        # store given values
        self.verbose = verbose
        # initialize internal attributes
        self.unique_classes = None
        self.aps = defaultdict(float)
        self.load_start = None
        self.load_end = None
        self.eval_start = None
        self.eval_end = None
        self.accm_start = None
        self.accm_end = None
        self._load(path_true=trues, path_pred=preds)
        return

    @staticmethod
    def _load_jsonl(
        path: str,
        model: BaseModel,
        uniq_classes: set
    ) -> Dict:
        ''' read json lines format file

        * read json lines format file
        * cast each entry (= image) to the given BaseModel class
        * regist class id of each bouding box to uniq_classes

        Args:
            path (str): path of json lines format file
            model (BaseModel): the BaseModel to cast image data
            uniq_classes (set): unique class ids of bounding box

        Returns:
            Dict[str, BaseModel]: dictionary of image_id(str) to BaseModel
        '''
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
                        uniq_classes.add(bbox.class_id)
                    entries[entry.image_id] = entry
                finally:
                    raw = rf.readline()
        return entries

    def _load(self: Evaluator, path_true: str, path_pred: str) -> None:
        '''load files and calc its basic values for evaluation

        * load json lines format file of ground truth and predictions
        * devide bouding boxes by class id for each image
        * call EvalBase.append (calc IoU and True-Positive and store them)

        Args:
            path_true (str):
                path of json lines format file of ground truths
            path_pred (str):
                path of json lines format file of predictions
        '''
        uniq_classes = set()
        self.load_start = time.perf_counter()
        # load json lines format file
        trues = self._load_jsonl(
            path=path_true, model=GroundTruth, uniq_classes=uniq_classes
        )
        preds = self._load_jsonl(
            path=path_pred, model=Prediction, uniq_classes=uniq_classes
        )
        self.load_end = time.perf_counter()
        self.unique_classes = sorted(list(uniq_classes))
        # check image_ids are unique in each data
        assert len(list(trues.keys())) == len(set(trues.keys()))
        assert len(list(preds.keys())) == len(set(preds.keys()))
        # create BaseEval for each class
        self.bases = dict()
        for class_id in self.unique_classes:
            self.bases[class_id] = BaseEval(class_id=class_id)
        self.eval_start = time.perf_counter()
        # get GroundTruth and Prediction which has the same image_id
        for image_id, pred in preds.items():
            true = trues.get(image_id)
            if true is None:
                continue
            # convert GroundTruth and Prediction to np.ndarray
            # pred is sorted in descending order of confidence score
            true_bboxes = true.to_ndarray()
            pred_bboxes = pred.to_ndarray()
            # divide bounding boxes by class_id
            for class_id in self.unique_classes:
                true_index = (true_bboxes[:, 4] == class_id)
                true_count = true_index.astype(int).sum()
                pred_index = (pred_bboxes[:, 5] == class_id)
                pred_count = pred_index.astype(int).sum()
                if true_count == 0 and pred_count == 0:
                    continue
                # calc IoU, True-Positive for each image and class
                self.bases[class_id].append(
                    true=true_bboxes[true_index][:, :-1],
                    pred=pred_bboxes[pred_index][:, :-1]
                )
        self.eval_end = time.perf_counter()
        return

    def accumulate(self: Evaluator) -> None:
        self.accm_start = time.perf_counter()
        for class_id in self.unique_classes:
            base = self.bases[class_id]
            if base.n_true == 0 and base.n_pred == 0:
                del self.bases[class_id]
            else:
                base.accumulate()
        aps_all = list()
        for i in range(10):
            th_ind = 50 + (i * 5)
            aps = list()
            for base in self.bases.values():
                aps.append(base.aps[th_ind])
            ap = np.array(aps).mean()
            self.aps[th_ind] = ap
            aps_all.append(ap)
        ap_all = np.array(aps_all).mean()
        self.aps[100] = ap_all
        self.accm_end = time.perf_counter()
        return

    def __str__(self: Evaluator) -> str:
        text = '===== mean Average Precision (mAP) =====\n'
        if self.load_start is None or self.load_end is None:
            text += 'Loading json lines format files is not done\n'
            return text
        if self.eval_start is None or self.eval_end is None:
            text += 'Evaluating bouding boxes is not done\n'
            return text
        if self.accm_start is None or self.accm_end is None:
            text += 'Accumlating evaluation result is not done\n'
            return text
        elapsed_load = (self.load_end - self.load_start) / 1000
        elapsed_eval = (self.eval_end - self.eval_start) / 1000
        elapsed_accm = (self.accm_end - self.accm_start) / 1000
        text += f'Loading bounding boxes: {elapsed_load:.3} sec\n'
        text += f'Evaluating bounding boxes: {elapsed_eval:.3} sec\n'
        text += f'Accumulating evaluation result: {elapsed_accm:.3} sec\n'
        text += '=== class: all ===\n'
        for i in range(11):
            th_ind = 100 - (i * 5)
            ap = self.aps[th_ind]
            if th_ind == 100:
                text += f'mAP @ [IoU=0.50:0.95] = {ap:<5.3}\n'
            else:
                text += f'mAP @ [IoU=0.{th_ind}     ] = {ap:<5.3}\n'
        if not self.verbose:
            return text
        for class_id, base in sorted(self.bases.items()):
            text += f'=== class: {class_id} ===\n'
            text += f'# of Ground Truth = {base.n_true}\n'
            text += f'# of Prediction   = {base.n_pred}\n'
            text += f'# of Image        = {base.n_img}\n'
            for th_ind in [100, 75, 50]:
                ap = base.aps[th_ind]
                if th_ind == 100:
                    text += f'mAP @ [IoU=0.50:0.95] = {ap:<5.3}\n'
                else:
                    text += f'mAP @ [IoU=0.{th_ind}     ] = {ap:<5.3}\n'
        return text


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
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='output mAP for each class'
    )
    args = parser.parse_args()
    evaluator = Evaluator(**vars(args))
    evaluator.accumulate()
    print(evaluator)
    return
