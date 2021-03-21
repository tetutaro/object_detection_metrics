#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from collections import defaultdict
import numpy as np


class Category(object):
    '''the basis of evaluation per category id

    Attributes:
        category_id (int): category id of detected object
        n_true (int): the number of ground truth bounding boxes
        n_pred (int): the number of predicted bounding boxes
        n_img (int): the number of images which has bouding box
                     of the category id (either ground truth or prediction)
        tps (Dict[int, List[np.ndarray]]):
            True-Positives (True-Positives and confidence score)
            per IoU threshold.
            the values of key are int(threshold * 100).
            value 100 of key means IoU=0.50:0.95:0.05
        aps (Dict[int, float]):
            Average Precisions per IoU threshold.
            the values of key are int(threshold * 100).
            value 100 of key means IoU=0.50:0.95:0.05
    '''
    def __init__(
        self: Category,
        category_id: int,
    ) -> None:
        '''initialize function of Category

        Args:
            category_id (int): category id of detected object
        '''
        # store given values
        self.category_id = category_id
        # initialize internal attributes
        self.n_true = 0
        self.n_pred = 0
        self.n_img = 0
        self.tps = defaultdict(list)
        self.aps = defaultdict(float)
        return

    @staticmethod
    def calc_iou(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
        '''calculate IoU (Intersection of Union)

        * calc IoU of bounding boxes (ground truth and prediction) at once
        * If number of prediction is N and number of ground truth is M,
          this function produces N x M matrix (IoU matrix)
        * bounding boxes must be written as (min_x, min_y, max_x, max_y)

        Args:
            true (np.ndarray): bounding boxes of ground truth (M x 4)
            pred (np.ndarray): bounding boxes of prediction (N x 4)

        Returns:
            np.ndarray: IoU matrix (N x M)
        '''
        assert len(true.shape) == len(pred.shape) == 2
        assert true.shape[1] == pred.shape[1] == 4
        # expand bouding boxes to N x M x 4
        ex_like = np.zeros((pred.shape[0], true.shape[0], pred.shape[1]))
        ex_true = np.full_like(
            ex_like, true[np.newaxis, :, :], dtype=np.float
        )
        ex_pred = np.full_like(
            ex_like, pred[:, np.newaxis, :], dtype=np.float
        )
        # calc the area of bouding boxes
        area_true = (
            ex_true[:, :, 2] - ex_true[:, :, 0]
        ) * (
            ex_true[:, :, 3] - ex_true[:, :, 1]
        )
        area_pred = (
            ex_pred[:, :, 2] - ex_pred[:, :, 0]
        ) * (
            ex_pred[:, :, 3] - ex_pred[:, :, 1]
        )
        # calc intersections between ground truths and predictions
        left_ups = np.maximum(ex_true[:, :, :2], ex_pred[:, :, :2])
        right_downs = np.minimum(ex_true[:, :, 2:], ex_pred[:, :, 2:])
        intersections = np.maximum(right_downs - left_ups, 0.0)
        # calc area of intersection and union
        area_inter = intersections[:, :, 0] * intersections[:, :, 1]
        area_union = area_true + area_pred - area_inter
        # calc IoU and return it
        return np.maximum(
            1.0 * area_inter / area_union,
            np.finfo(np.float).eps
        )

    @staticmethod
    def calc_tp(iou: np.ndarray, threshold: float) -> np.ndarray:
        '''calculate True-Positives from IoU matrix

        * iou matrix represents IoU between ground truths and predictions
        * the shape of iou matrix (N x M) shows
          # of prediction is N and # of ground truth is M
        * predictions of iou matrix has been sorted
          in descending order of confidence score
        * this function produces N x 1 matrix and its values are 0 or 1
        * 1 means that its prediction is True-Positive
          and 0 means that its prediction is False-Positive
        * threshold is the minimum IoU that
          the prediction considers to be true
        * the ground truth is assigned from the prediction
          which has higher confidence score
        * the ground truth which has the highest IoU (and >= threshold)
          among all (the rest) ground truths is assigned to the prediction
        * the ground truth once assigned to the prediction is not assigned
          to other predictions
        * therefore, sum of # of True-Positives is less than or equal to
          # of ground truths

        Args:
            iou (np.ndarray): IoU matrix (N x M)
            threshold (float): IoU threshold

        Returns:
            np.ndarray: True-Positives (N x 1)
        '''
        masked = np.where(iou >= threshold, iou, 0)
        for i in range(iou.shape[0]):
            if masked[i, :].max() <= 0:
                continue
            ind = np.argmax(masked[i, :])
            masked[i, :] = 0
            masked[:, ind] = 0
            masked[i, ind] = 1
        return np.where(masked > 0, 1, 0).sum(axis=1)[:, np.newaxis]

    def append(
        self: Category,
        true: np.ndarray,
        pred: np.ndarray
    ) -> None:
        '''calc the basis of evaluation for each category id

        * store # of ground truths, predictions
        * calc IoU between ground truths and predictions
        * calc True-Positives for each IoU threshold
        * store it to self.tps
        * bouding boxes of prediction are sorted
          in descending order of confidence score
          at Prediction.to_ndarray()

        Args:
            true (np.ndarray): bounding boxes of ground truth (M x 4)
            pred (np.ndarray):
                bounding boxes of prediction (N x 5)
                (pred[:, 4] is a list of confidence scores)
        '''
        assert true.shape[0] > 0 or pred.shape[0] > 0
        # store # of ground truths, predictions
        self.n_true += true.shape[0]
        self.n_pred += pred.shape[0]
        self.n_img += 1
        # if # of predictions == 0, just count up # of ground truth
        if pred.shape[0] == 0:
            return
        # if # of ground truth == 0, all predictions are False-Positive
        if true.shape[0] == 0:
            for i in range(10):
                th_ind = 50 + (i * 5)
                tp = np.zeros((pred.shape[0], 1), dtype=np.int)
                tp = np.concatenate([tp, pred[:, 4:5]], axis=1)
                self.tps[th_ind].append(tp)
            return
        # calc IoU between ground truths and predictions
        iou = self.calc_iou(true=true[:, :4], pred=pred[:, :4])
        # calc True-Positives for each IoU threshold
        for i in range(10):
            threshold = 0.5 + (i * 0.05)
            th_ind = 50 + (i * 5)
            # calc True-Positives at the IoU threshold
            tp = self.calc_tp(iou=iou, threshold=threshold)
            ntp = tp.sum()
            assert ntp <= true.shape[0]
            # unite True-Positives and confidence score
            tp = np.concatenate([tp, pred[:, 4:5]], axis=1)
            # store it
            self.tps[th_ind].append(tp)
        return

    @staticmethod
    def calc_auc(x: np.ndarray, y: np.ndarray) -> float:
        '''calculate the area of interpolated curve

        Args:
            x (np.ndarray):
                x-axis of interpolated curve.
                to calc Average Precision, x is Recall.
                to calc Average Recall, x is Precision.
            y (np.ndarray):
                y-axis of interpolated curve.
                to calc Average Precision, y is Precision.
                to calc Average Recall, y is Recall.
        '''
        area_points = list()
        tmp_points = list(zip(x, y))
        key_point = tmp_points[0]
        # select the points to calc the area(area_points) from all points
        # == interpolating the Precision-Recall/Recall-Precision curve
        if len(tmp_points) == 1:
            area_points.append(key_point)
        else:
            for i, tmp_point in enumerate(tmp_points[1:]):
                if tmp_point[1] > key_point[1]:
                    # tmp_y > key_y
                    if tmp_point[0] < key_point[0]:
                        # tmp_x < key_x and tmp_y > key_y
                        # add key_point
                        area_points.append(key_point)
                    # update
                    key_point = tmp_point
                if i == len(tmp_points) - 2:
                    # the last tmp_point
                    # add key_point
                    area_points.append(key_point)
        # calc the area under the interpolated curve
        auc = 0
        base_x = 0
        for area_point in area_points[::-1]:
            auc += (area_point[0] - base_x) * area_point[1]
            base_x = area_point[0]
        return auc

    def calc_ap(self: Category, tps: np.ndarray) -> float:
        '''calculate Average Precision

        Args:
            tps (np.ndarray):
                True-Positives and confidence score (N x 2).
                tps[:, 0] is True-Positive.
                tps[:, 1] is confidence score.
                it is just a concatenated value from multiple images
                and so it has not been sorted by confidence score.

        Returns:
            float: Average Precision
        '''
        # sort in descending order of confidence score
        tps = tps[np.argsort(tps[:, 1])[::-1]]
        # calc accumulated True-Positives
        acc_tp = np.cumsum(tps[:, 0])
        # calc Precision and Recall
        precision = acc_tp / np.array(list(range(1, self.n_pred + 1)))
        recall = acc_tp / self.n_true
        # calc Average Precision and Average Recall
        ap = self.calc_auc(x=recall[::-1], y=precision[::-1])
        return ap

    def accumulate(self: Category) -> None:
        '''calc Average Precision of each IoU threshold for each category id

        * calc Average Precisions for each IoU threshold
        * store it to self.aps
        * calc Average Precision of IoU=0.50:0.95:0.05
        * store it to self.aps, too
        '''
        # if # of ground truths == 0, Average Precision is 0 obviously
        if self.n_true == 0:
            for i in range(10):
                th_ind = 50 + (i * 5)
                self.aps[th_ind] = 0.0
            self.aps[100] = 0.0
            return
        aps_all = list()
        # calc Average Precision for each IoU threshold
        for i in range(10):
            # get tps(True-Positives and confidence score)
            th_ind = 50 + (i * 5)
            tps = self.tps[th_ind]
            # if # of predictions == 0, Average Precision is 0 obviously
            if len(tps) == 0:
                self.aps[th_ind] = 0.0
                aps_all.append(0.0)
                continue
            # unite tps(True-Positives and confidence score) of all images
            tps = np.concatenate(tps, axis=0)
            # calc Average Precision
            ap = self.calc_ap(tps=tps)
            # store it
            self.aps[th_ind] = ap
            aps_all.append(ap)
        # calc Average Precision of IoU=0.50:0.95:0.05 and store it
        self.aps[100] = np.array(aps_all).mean()
        return
