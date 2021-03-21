#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Union, Optional
import os
from collections import defaultdict
import simplejson as json
import argparse


class Converter(object):
    def __init__(
        self: Converter,
        coco: str,
        output: Optional[str]
    ) -> None:
        if not os.path.isfile(coco):
            raise ValueError(f'coco({coco}) not found')
        if not coco.endswith('.json'):
            raise ValueError('coco must be COCO annotation json file')
        outbase = os.path.basename(coco).replace('.json', '.jsonl')
        if output is not None:
            if output.endswith('.jsonl'):
                self.output = output
            else:
                self.output = os.path.join(output, outbase)
        else:
            self.output = outbase
        with open(coco, 'rt') as rf:
            anns = json.load(rf)
        self.bboxes = defaultdict(list)
        if isinstance(anns, dict):
            # ground truth
            for ann in anns['annotations']:
                image_id = '%012d' % ann['image_id']
                self.bboxes[image_id].append({
                    'category_id': ann['category_id'],
                    'bbox': self.convert_xywh_xyxy(ann['bbox']),
                })
        elif isinstance(anns, list):
            # (fake) prediction result
            for ann in anns:
                image_id = '%012d' % ann['image_id']
                self.bboxes[image_id].append({
                    'category_id': ann['category_id'],
                    'bbox': self.convert_xywh_xyxy(ann['bbox']),
                    'score': ann['score'],
                })
        else:
            raise ValueError('invalid file type')
        return

    @staticmethod
    def convert_xywh_xyxy(
        xywh: List[Union[int, float]]
    ) -> List[float]:
        return [
            float(xywh[0]),
            float(xywh[1]),
            float(xywh[0] + xywh[2]),
            float(xywh[1] + xywh[3])
        ]

    def save(self: Converter) -> None:
        with open(self.output, 'wt') as wf:
            for image_id, bboxes in sorted(self.bboxes.items()):
                wf.write(json.dumps({
                    'image_id': image_id,
                    'bboxes': bboxes,
                }) + '\n')
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description='convert coco annotations to jsonl format',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'coco', type=str,
        help='coco annotation file'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help=(
            'output jsonl file\n'
            "(default: os.path.basename(coco).replace('.json', '.jsonl')"
        )
    )
    args = parser.parse_args()
    converter = Converter(**vars(args))
    converter.save()
    return


if __name__ == '__main__':
    main()
