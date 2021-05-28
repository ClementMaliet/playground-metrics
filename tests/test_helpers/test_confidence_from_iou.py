# flake8: noqa: E501
import numpy as np
from pygeos import box

from playground_metrics.helpers import add_confidence_from_max_iou


def test_confidence_from_max_iou_bbox():
    detections = np.array([[box(0, 0, 9, 5)],
                           [box(23, 13, 29, 18)]])
    ground_truths = np.array([[box(5, 2, 15, 9)],
                              [box(18, 10, 26, 15)]])
    res = add_confidence_from_max_iou(detections, ground_truths)

    assert np.all(res == np.array([[box(0, 0, 9, 5), 0.11650485436893204],
                                   [box(23, 13, 29, 18), 0.09375]],
                                  dtype=object))
