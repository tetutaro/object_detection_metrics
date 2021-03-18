#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
import numpy as np


class BaseEval(object):
    def __init__(
        self: BaseEval,
        image_id: str,
        class_id: int,
        true: np.ndarray,
        pred: np.ndarray
    ) -> None:
        self.image_id = image_id
        self.class_id = class_id
        self.true = true
        self.pred = pred
        return

    def __str__(self: BaseEval) -> str:
        text = 'CLASS: ' + str(self.class_id) + '\n'
        text += 'IMAGE: ' + self.image_id + '\n'
        text += 'TRUE:\n' + str(self.true) + '\n'
        text += 'PRED:\n' + str(self.pred)
        return text
