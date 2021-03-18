#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List
from setuptools import setup
import object_detection_metrics


def read_requirements() -> List[str]:
    with open('requirements.txt', 'rt') as rf:
        required = rf.read().strip().splitlines()
    return required


setup(
    name='object_detection_metrics',
    version=object_detection_metrics.__version__,
    license='MIT',
    description='calculate metrics used to object detection algorythm',
    author='tetutaro',
    author_email='tetsutaro.maruyama@gmail.com',
    url='https://github.com/tetutaro/object_detection_metrics',
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'object_detection_metrics = object_detection_metrics:main',
        ],
    },
    packages=[
        'object_detection_metrics',
    ]
)
