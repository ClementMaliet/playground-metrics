__version__ = '1.0.0'

from .map_metric import MeanAveragePrecisionMetric
from .utils import get_type_and_convert, convert_to_bounding_box, convert_to_point, convert_to_polygon
from .match import non_unitary_match, coco_match, xview_match, IntersectionOverUnionMatcher, EuclideanMatcher, \
    PointInBoxMatcher, ConstantBoxMatcher
