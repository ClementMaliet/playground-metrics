"""Implement the public interface to match a set of detections and ground truths."""

from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
from pygeos import area, is_empty, intersection

from .functional import non_unitary_match, xview_match, coco_match, sort_detection_by_confidence
from .geometry import GeometryType, intersection_over_union, is_type, euclidean_distance, as_boxes, \
    point_to_box, as_points


class _FunctionValue:
    """Used to make a useless wrapper around functions to allow them to pass as enumeration values."""

    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class MatchAlgorithm(Enum):
    """Every available match algorithms by name."""

    coco = _FunctionValue(coco_match)
    xview = _FunctionValue(xview_match)
    non_unitary = _FunctionValue(non_unitary_match)

    def __call__(self, *args, **kwargs):
        r"""Match geometries for a given similarity matrix and delta matrix.

        Args:
            similarity_matrix (numpy.ndarray): A (#detection, #gt) similarity matrix between detections and
                ground truths.
            delta_matrix (numpy.ndarray): A binary (#detection, #gt) matrix which encodes the trim step results.

        Returns:
            numpy.ndarray: A binary matrix of all matches of dimension (#detections, #ground truth)

        """
        return self.value(*args, **kwargs)


class MatchEngine(ABC):
    """Match detection with their ground truth according a similarity matrix and a detection confidence score.

    Matching may be done using coco algorithm or xView algorithm (which yield different matches as described for an
    intersection-over-union similarity matrix in :ref:`match`) or with non-unitary matching.

    Subclasses must implement :meth:`compute_similarity_matrix` and :meth:`trim_similarity_matrix` to be functional.

    Args:
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' to choose the match algorithm

    Attributes:
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' and indicates the match algorithm used

    """

    def __init__(self, match_algorithm):
        if match_algorithm not in ['coco', 'xview', 'non-unitary']:
            raise ValueError("match_algorithm must be either coco, xview or non-unitary")

        self.match_algorithm = match_algorithm

        # Authorized geometric types fot this match engine
        self._detection_types = (GeometryType.POLYGON, GeometryType.POINT)
        self._ground_truth_types = (GeometryType.POLYGON, GeometryType.POINT)

    def __repr__(self):
        """Represent the :class:`~playground_metrics.match_detections.MatchEngine` as a string."""
        d_arg = []
        for arg in ['threshold', 'match_algorithm', 'bounding_box_size']:
            if hasattr(self, arg):
                d_arg.append('{}={}'.format(arg, self.__getattribute__(arg)))
        return '{}({})'.format(self.__class__.__name__, ', '.join(d_arg))

    def __str__(self):
        """Represent the :class:`~playground_metrics.match_detections.MatchEngine` as a string."""
        d_arg = []
        for arg in ['threshold', 'match_algorithm', 'bounding_box_size']:
            if hasattr(self, arg):
                d_arg.append('{}={}'.format(arg, self.__getattribute__(arg)))
        return '{}({})'.format(self.__class__.__name__.replace('MatchEngine', ''), ', '.join(d_arg))

    def _compute_similarity_matrix_and_trim(self, detections, ground_truths, label_mean_area=None):
        similarity_matrix = self.compute_similarity_matrix(detections, ground_truths, label_mean_area)
        return similarity_matrix, self.trim_similarity_matrix(similarity_matrix, detections, ground_truths,
                                                              label_mean_area)

    @abstractmethod
    def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):
        r"""Compute a similarity matrix between detections and ground truths.

        Abstract method.

        This method must be overridden in subsequent subclasses to handle both bounding box and polygon input format.

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray : A similarity matrix of dimension (#detections, #ground truth)

        """
        raise NotImplementedError

    @abstractmethod
    def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths,
                               label_mean_area=None):  # noqa: D205,D400
        r"""Compute an array containing the indices in columns of similarity passing the first trimming (typically for
        IoU this would be the result of a simple thresholding) but it might be any method fit to do a rough filtering of
        possible ground truth candidates to match with a given detection.

        Abstract method.

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray: An array of dimension (2, N) where each column is a tuple (detection, ground truth) describing
            a potential match. To be more precise, each match-tuple in the array corresponds to a position in the
            similarity matrix which will be used by the match algorithm to compute the final match.

        """
        raise NotImplementedError

    def match(self, detections, ground_truths, label_mean_area=None):  # noqa: D205,D400
        r"""Match detections :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` with ground truth
        :class:`~playground_metrics.utils.geometry_utils.geometry.Geometry` at a given similarity matrix and trim
        method using either Coco algorithm, xView algorithm or a naive *non-unitary* match.

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray : A binary matrix of all matches of dimension (#detections, #ground truth)

        """
        if detections.shape[0] == 0:
            return np.zeros((0, ground_truths.shape[0]))

        if ground_truths.shape[0] == 0:
            return np.zeros((detections.shape[0], 0))

        # Geometric static typing
        if not np.all(is_type(detections[:, 0], *self._detection_types)):
            raise TypeError('Invalid geometric type provided in '
                            'detections, expected to be on of {}'
                            ''.format(' '.join(['{}'.format(geom_type.name)
                                                for geom_type in self._detection_types])))
        if not np.all(is_type(ground_truths[:, 0], *self._ground_truth_types)):
            raise TypeError('Invalid geometric type provided in '
                            'detections, expected to be on of {}'
                            ''.format(' '.join(['{}'.format(geom_type.name)
                                                for geom_type in self._ground_truth_types])))

        # We sort detections by confidence before computing the similarity matrix
        detections = sort_detection_by_confidence(detections)

        # Compute similarity matrix and An array containing the indices in columns of similarity passing the first
        # trimming (Typically for IoU this would be the result of a simple thresholding).
        similarity_matrix, delta_matrix = self._compute_similarity_matrix_and_trim(detections,
                                                                                   ground_truths,
                                                                                   label_mean_area)

        # We match the detection and the ground truth using the configured algorithm
        try:
            return MatchAlgorithm[self.match_algorithm.replace('-', '_')](similarity_matrix, delta_matrix)
        except KeyError as error:
            raise ValueError('Invalid match algorithm: '
                             'Expected one of ({})'.format(', '.join(MatchAlgorithm.__members__.keys()))) from error


class MatchEngineIoU(MatchEngine):
    """Match detection with their ground truth according the their IoU and the detection confidence score.

    Args:
        threshold (float): The IoU threshold at which one considers a potential match as valid
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' to choose the match algorithm

    """

    def __init__(self, threshold, match_algorithm):
        super(MatchEngineIoU, self).__init__(match_algorithm)

        self._detection_types = (GeometryType.POLYGON, )
        self._ground_truth_types = (GeometryType.POLYGON, )

        self.threshold = threshold

    def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):
        r"""Compute the iou scores between all pairs of geometries with an Rtree on detections to speed up computation.

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset,
                if given, it is used to match with *iIoU* instead of *IoU* (c.f. :ref:`iiou`)

        Returns:
            ndarray : An IoU matrix (#detections, #ground truth)

        """
        detections = sort_detection_by_confidence(detections)
        iou = intersection_over_union(detections[:, 0], ground_truths[:, 0])
        if label_mean_area is not None:
            iou = (label_mean_area / area(ground_truths[:, 0])) * iou

        return iou

    def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths, label_mean_area=None):
        r"""Compute an array containing the indices in columns of similarity passing the first trimming.

        Here this is the result of a simple thresholding over IoU.

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset,
                if given, it is used to match with *iIoU* instead of *IoU* (c.f. :ref:`iiou`)

        Returns:
            ndarray: An array of dimension (2, N) where each column is a tuple (detection, ground truth) describing
            a potential match. To be more precise, each match-tuple in the array corresponds to a position in the
            similarity matrix which will be used by the match algorithm to compute the final match.

        """
        res = np.stack(np.nonzero(similarity_matrix >= self.threshold))
        return res[:, np.argsort(np.nonzero(similarity_matrix >= self.threshold)[0])]


class MatchEngineEuclideanDistance(MatchEngine):
    """Match detection with their ground truth according the their relative distance and the detection confidence score.

    Args:
        threshold (float): The distance threshold at which one considers a potential match as valid
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' to choose the match algorithm

    """

    def __init__(self, threshold, match_algorithm):
        super(MatchEngineEuclideanDistance, self).__init__(match_algorithm)
        self._threshold = 1 - threshold

    @property
    def threshold(self):
        """float: The distance threshold at which one considers a potential match as valid."""
        return 1 - self._threshold

    def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):
        r"""Compute a partial similarity matrix based on the euclidean distance between all pairs of points.

        The difference with :class:`~playground_metrics.match_detections.MatchEnginePointInBox` lies in the
        similarity matrix rough trimming which depends on a threshold rather than on whether a detection (as a point)
        lies within a ground truth polygon (or bounding box).

        The computed matrix is :math:`\mathcal{S} = 1 - \mathcal{D}` with:

        .. math::

            \mathcal{D}_{ij} = \begin{cases} \left\lVert d_i - gt_i \right\rVert_2 &\mbox{if } d_i \in B(gt_i, t)\\
                \inf &\mbox{if }  d_i \notin B(gt_i, t) \end{cases}

        Where :math:`B(gt_i, t)` is a square box centered in :math:`gt_i` of size length :math:`t`.

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray : An similarity matrix (#detections, #ground truth)

        """
        detections = sort_detection_by_confidence(detections)
        similarity = euclidean_distance(as_points(detections[:, 0]),
                                        point_to_box(ground_truths[:, 0],
                                                     width=2 * self.threshold,
                                                     height=2 * self.threshold))
        return similarity

    def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths, label_mean_area=None):
        r"""Compute an array containing the indices in columns of similarity passing the first trimming.

        Here this is the result of a simple thresholding over the distance matrix.

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray: An array of dimension (2, N) where each column is a tuple (detection, ground truth) describing
            a potential match. To be more precise, each match-tuple in the array corresponds to a position in the
            similarity matrix which will be used by the match algorithm to compute the final match.

        """
        res = np.stack(np.nonzero(similarity_matrix >= self._threshold))
        return res[:, np.argsort(np.nonzero(similarity_matrix >= self._threshold)[0])]


class MatchEnginePointInBox(MatchEngine):  # noqa: D205,D400
    """Match detection with their ground truth according the their relative distance, whether a detection point is in a
    ground truth box and the detection confidence score.

    Args:
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' to choose the match algorithm

    """

    def __init__(self, match_algorithm):
        super(MatchEnginePointInBox, self).__init__(match_algorithm)

        self._ground_truth_types = (GeometryType.POLYGON, )

    def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):  # noqa: D205,D400
        r"""Compute a partial similarity matrix based on the euclidean distance between all pairs of points with an
        Rtree on detections to speed up computation.

        The difference with :class:`~playground_metrics.match_detections.MatchEngineEuclideanDistance` lies in the
        similarity matrix rough trimming which depends on whether a detection (as a point) lies within a ground truth
        polygon (or bounding box) rather than on a threshold.

        The computed matrix is :math:`\mathcal{S} = 1 - \mathcal{D}` with:

        .. math::

            \mathcal{D}_{ij} = \left\lVert d_i - gt_i \right\rVert_2

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray : An similarity matrix (#detections, #ground truth)

        """
        detections = sort_detection_by_confidence(detections)
        similarity = euclidean_distance(as_points(detections[:, 0]),
                                        as_boxes(ground_truths[:, 0]))
        return similarity

    def trim_similarity_matrix(self, similarity_matrix, detections, ground_truths, label_mean_area=None):
        r"""Compute an array containing the indices in columns of similarity passing the first trimming.

        Here a detection/ground truth pair is kept if the detection
        :class:`~playground_metrics.utils.geometry_utils.geometry.Point` is within the ground truth
        :class:`~playground_metrics.utils.geometry_utils.geometry.BoundingBox` or
        :class:`~playground_metrics.utils.geometry_utils.geometry.Polygon`

        Args:
            similarity_matrix: The similarity matrix between detections and ground truths : dimension (#detection, #gt)
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray: An array of dimension (2, N) where each column is a tuple (detection, ground truth) describing
            a potential match. To be more precise, each match-tuple in the array corresponds to a position in the
            similarity matrix which will be used by the match algorithm to compute the final match.

        """
        potential = np.stack(np.nonzero(similarity_matrix != -np.Inf))
        potential = potential[:, np.argsort(np.nonzero(similarity_matrix != -np.Inf)[0])]

        trim = []
        for i in range(potential.shape[1]):
            r, c = potential[:, i]
            if np.all(is_empty(intersection(detections[r, 0], ground_truths[c, 0]))):
                trim.append(i)

        return np.delete(potential, trim, axis=1)


class MatchEngineConstantBox(MatchEngineIoU):  # noqa: D205,D400
    """Match detection with their ground truth according the IoU computed on fixed-size
    bounding boxes around detection and ground truth points and the detection confidence score.

    Args:
        threshold (float): The IoU threshold at which one considers a potential match as valid
        match_algorithm (str) : Either 'coco', 'xview' or 'non-unitary' to choose the match algorithm
        bounding_box_size (float): The fixed-size bounding box size

    """

    def __init__(self, threshold, match_algorithm, bounding_box_size):
        super(MatchEngineConstantBox, self).__init__(threshold, match_algorithm)
        self.bounding_box_size = bounding_box_size

        # Override authorized geometric types fot this match engine
        self._detection_types = (GeometryType.POLYGON, GeometryType.POINT)
        self._ground_truth_types = (GeometryType.POLYGON, GeometryType.POINT)

    def compute_similarity_matrix(self, detections, ground_truths, label_mean_area=None):  # noqa: D205,D400
        r"""Compute a parial similarity matrix based on the intersection-over-union between all pairs of constant-sized
        bounding box around points with an Rtree on detections to speed up computation.

        Args:
            detections (ndarray, list) : A ndarray of detections stored as:

                * Bounding boxes for a given class where each row is a detection stored as:
                  ``[BoundingBox, confidence]``
                * Polygons for a given class where each row is a detection stored as:
                  ``[Polygon, confidence]``
                * Points for a given class where each row is a detection stored as:
                  ``[Point, confidence]``

            ground_truths (ndarray,list) : A ndarray of ground truth stored as:

                * Bounding boxes for a given class where each row is a ground truth stored as:
                  ``[BoundingBox]``
                * Polygons for a given class where each row is a ground truth stored as:
                  ``[Polygon]``
                * Points for a given class where each row is a ground truth stored as:
                  ``[Point]``

            label_mean_area (float) : Optional, default to ``None``. The mean area for each label in the dataset.

        Returns:
            ndarray : An IoU matrix (#detections, #ground truth)

        """
        detections = np.stack((point_to_box(detections[:, 0],
                                            width=self.bounding_box_size,
                                            height=self.bounding_box_size),
                               detections[:, 1]), axis=1)
        ground_truths = point_to_box(ground_truths[:, 0],
                                     width=self.bounding_box_size,
                                     height=self.bounding_box_size)[:, None]

        return super(MatchEngineConstantBox, self).compute_similarity_matrix(detections,
                                                                             ground_truths,
                                                                             label_mean_area)
