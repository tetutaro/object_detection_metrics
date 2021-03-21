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
from .category import Category, CategoryTotal
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
        categories (Dict[int, Category]):
            store Category class
        unique_categories (List[int]):
            unique category ids of detected objects.
            it may have unused category ids because
            object detection may not be made on
            all images of ground truths.
        macro_maps (Dict[int, float]):
            macro mean Average Precisions per IoU threshold.
            the values of key are int(threshold * 100).
            value 100 of key means IoU=0.50:0.95:0.05
        weighted_maps (Dict[int, float]):
            weighted mean Average Precisions per IoU threshold.
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
        self.categories = dict()
        self.unique_categories = None
        self.category_total = CategoryTotal()
        self.macro_maps = defaultdict(float)
        self.weighted_maps = defaultdict(float)
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
        uniq_categories: set
    ) -> Dict:
        ''' read json lines format file

        * read json lines format file
        * cast each entry (= image) to the given BaseModel class
        * regist category id of each bouding box to uniq_categories

        Args:
            path (str): path of json lines format file
            model (BaseModel): the BaseModel to cast image data
            uniq_categories (set): unique category ids of bounding box

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
                        uniq_categories.add(bbox.category_id)
                    entries[entry.image_id] = entry
                finally:
                    raw = rf.readline()
        return entries

    def _load(self: Evaluator, path_true: str, path_pred: str) -> None:
        '''load files and calc its basic values for evaluation

        * load json lines format file of ground truth and predictions
        * devide bouding boxes by category id for each image
        * call EvalBase.append (calc IoU and True-Positive and store them)

        Args:
            path_true (str):
                path of json lines format file of ground truths
            path_pred (str):
                path of json lines format file of predictions
        '''
        uniq_categories = set()
        self.load_start = time.perf_counter()
        # load json lines format file
        trues = self._load_jsonl(
            path=path_true, model=GroundTruth, uniq_categories=uniq_categories
        )
        preds = self._load_jsonl(
            path=path_pred, model=Prediction, uniq_categories=uniq_categories
        )
        self.load_end = time.perf_counter()
        self.unique_categories = sorted(list(uniq_categories))
        # check image_ids are unique in each data
        assert len(list(trues.keys())) == len(set(trues.keys()))
        assert len(list(preds.keys())) == len(set(preds.keys()))
        # create Category class for each category
        for category_id in self.unique_categories:
            self.categories[category_id] = Category(category_id=category_id)
        self.eval_start = time.perf_counter()
        # get GroundTruth and Prediction which has the same image_id
        for image_id, pred in preds.items():
            true = trues.get(image_id)
            if true is None:
                continue
            self.category_total.n_img += 1
            # convert GroundTruth and Prediction to np.ndarray
            # pred is sorted in descending order of confidence score
            true_bboxes = true.to_ndarray()
            pred_bboxes = pred.to_ndarray()
            # divide bounding boxes by category_id
            for category_id in self.unique_categories:
                true_index = (true_bboxes[:, 4] == category_id)
                true_count = true_index.astype(int).sum()
                pred_index = (pred_bboxes[:, 5] == category_id)
                pred_count = pred_index.astype(int).sum()
                if true_count == 0 and pred_count == 0:
                    continue
                # calc IoU, True-Positive for each image and category
                self.categories[category_id].append(
                    true=true_bboxes[true_index][:, :-1],
                    pred=pred_bboxes[pred_index][:, :-1]
                )
        self.eval_end = time.perf_counter()
        return

    def accumulate(self: Evaluator) -> None:
        '''accumulate all Average Precisions
        '''
        self.accm_start = time.perf_counter()
        for category_id in self.unique_categories:
            category = self.categories[category_id]
            if category.n_true == 0 and category.n_pred == 0:
                # delete unused category id
                del self.categories[category_id]
            else:
                # calc Average Precision for each category
                category.accumulate()
                # copy each category's data to category_total
                self.category_total.n_true += category.n_true
                self.category_total.n_pred += category.n_pred
                for i in range(10):
                    th_ind = 50 + (i * 5)
                    tps = category.tps[th_ind]
                    if len(tps) == 0:
                        continue
                    self.category_total.tps[th_ind].append(
                        np.concatenate(tps, axis=0)
                    )
        # calc micro mean Average Precisions
        self.category_total.accumulate()
        # calc macro & weighted mean Average Precisions
        macro_maps_all = list()
        weighted_maps_all = list()
        for i in range(10):
            th_ind = 50 + (i * 5)
            trues = list()
            aps = list()
            for category in self.categories.values():
                trues.append(category.n_true)
                aps.append(category.aps[th_ind])
            trues = np.array(trues)
            aps = np.array(aps)
            macro_map = aps.mean()
            weighted_map = (trues * aps).sum() / trues.sum()
            self.macro_maps[th_ind] = macro_map
            self.weighted_maps[th_ind] = weighted_map
            macro_maps_all.append(macro_map)
            weighted_maps_all.append(weighted_map)
        self.macro_maps[100] = np.array(macro_maps_all).mean()
        self.weighted_maps[100] = np.array(weighted_maps_all).mean()
        self.accm_end = time.perf_counter()
        return

    @staticmethod
    def print_metrics(metrics: Dict, name: str, is_full: bool) -> str:
        '''show metrics for each IoU thresholf
        '''
        text = ''
        if is_full:
            threads = [100 - (i * 5) for i in range(11)]
        else:
            threads = [100, 75, 50]
        for th_ind in threads:
            val = metrics[th_ind]
            if th_ind == 100:
                text += f'{name} @ [IoU=0.50:0.95] = {val:<5.3}\n'
            else:
                text += f'{name} @ [IoU=0.{th_ind}     ] = {val:<5.3}\n'
        return text

    def __str__(self: Evaluator) -> str:
        '''show all the evaluated metrics

        Returns:
            str: text to print out
        '''
        text = '===== Evaluation Results =====\n'
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
        text += f'# of Image        = {self.category_total.n_img}\n'
        text += f'# of Ground Truth = {self.category_total.n_true}\n'
        text += f'# of Prediction   = {self.category_total.n_pred}\n'
        text += '===== mean Average Precision (mAP) =====\n'
        text += self.print_metrics(
            metrics=self.category_total.aps,
            name='micro mAP',
            is_full=self.verbose
        )
        text += self.print_metrics(
            metrics=self.macro_maps,
            name='macro mAP',
            is_full=self.verbose
        )
        text += self.print_metrics(
            metrics=self.weighted_maps,
            name='weighted mAP',
            is_full=self.verbose
        )
        if not self.verbose:
            return text
        for category_id, category in sorted(self.categories.items()):
            text += f'=== category: {category_id} ===\n'
            text += f'# of Image        = {category.n_img}\n'
            text += f'# of Ground Truth = {category.n_true}\n'
            text += f'# of Prediction   = {category.n_pred}\n'
            text += self.print_metrics(
                metrics=category.aps,
                name='AP',
                is_full=False
            )
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
        help='output mAP for each category'
    )
    args = parser.parse_args()
    evaluator = Evaluator(**vars(args))
    evaluator.accumulate()
    print(evaluator)
    return
