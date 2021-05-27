import numpy as np


def non_unitary_match(similarity_matrix, delta_matrix):
    r"""Match geometries for a given similarity matrix for every positive example yielded by the delta matrix.

    Args:
        similarity_matrix (numpy.ndarray): A (#detection, #gt) similarity matrix between detections and ground truths.
        delta_matrix (numpy.ndarray): A binary (#detection, #gt) matrix which encodes the trim step results.

    Returns:
        numpy.ndarray: A binary matrix of all matches of dimension (#detections, #ground truth)

    """
    # We prepare the detection match matrix
    match_matrix = np.zeros_like(similarity_matrix)
    match_matrix[delta_matrix[0, :], delta_matrix[1, :]] = 1

    return match_matrix


def xview_match(similarity_matrix, delta_matrix):
    r"""Match geometries for a given similarity matrix and delta matrix using xView algorithm.

    Args:
        similarity_matrix (numpy.ndarray): A (#detection, #gt) similarity matrix between detections and ground truths.
        delta_matrix (numpy.ndarray): A binary (#detection, #gt) matrix which encodes the trim step results.

    Returns:
        numpy.ndarray: A binary matrix of all matches of dimension (#detections, #ground truth)

    """
    # We prepare the detection match matrix
    match_matrix = np.zeros_like(similarity_matrix)

    if delta_matrix.shape[1] == 0:  # No matches at all
        return match_matrix

    ground_truth_match_vector = [0] * similarity_matrix.shape[1]

    forward = {match[0, 0]: match[1, :]
               for match in np.hsplit(delta_matrix, np.where(np.diff(delta_matrix[0, :]) != 0)[0] + 1)}

    for k in range(similarity_matrix.shape[0]):
        # For each detection we select its ground truth match
        detection_matches = forward.get(k, np.zeros((0, 0)))

        # If we don't have anything left to match -> skip
        if detection_matches.size == 0:
            continue

        # We select the biggest similarity_matrix over them
        m = np.argmax(similarity_matrix[k, detection_matches])
        n = detection_matches[m]

        if ground_truth_match_vector[n] == 0:
            # We match the detection and the ground truth
            ground_truth_match_vector[n] = 1
            match_matrix[k, n] = 1

    return match_matrix


def coco_match(similarity_matrix, delta_matrix):
    r"""Match geometries for a given similarity matrix and trim method using Coco algorithm.

    Args:
        similarity_matrix (numpy.ndarray): A (#detection, #gt) similarity matrix between detections and ground truths.
        delta_matrix (numpy.ndarray): A binary (#detection, #gt) matrix which encodes the trim step results.

    Returns:
        numpy.ndarray: A binary matrix of all matches of dimension (#detections, #ground truth)

    """
    # We prepare the detection match matrix
    match_matrix = np.zeros_like(similarity_matrix)

    if delta_matrix.shape[1] == 0:  # No matches at all
        return match_matrix

    forward = {match[0, 0]: match[1, :]
               for match in np.hsplit(delta_matrix, np.where(np.diff(delta_matrix[0, :]) != 0)[0] + 1)}
    similarity_matches_by_gt = delta_matrix[:, np.argsort(delta_matrix[1, :])]
    backward = {match[1, 0]: match[0, :]
                for match in np.hsplit(similarity_matches_by_gt,
                                       np.where(np.diff(similarity_matches_by_gt[1, :]) != 0)[0] + 1)}

    for k in range(similarity_matrix.shape[0]):
        # For each detection we select its ground truth match
        detection_matches = forward.get(k, np.zeros((0, 0)))

        # If we don't have anything left to match -> skip
        if detection_matches.size == 0:
            continue

        # We select the biggest similarity_matrix over them
        m = np.argmax(similarity_matrix[k, detection_matches])
        n = detection_matches[m]

        # We delete the ground truth column index from future match testing
        for d in backward[n]:
            forward[d] = forward[d][forward[d] != n]

        # We set the match flag to 1
        match_matrix[k, n] = 1

    return match_matrix


def sort_detection_by_confidence(detections):
    """Sort an array of detection by decreasing confidence.

    Args:
        detections (numpy.ndarray): An array of ``[pygeos.Geometry, confidence, label]`` detections.

    Returns:
        numpy.ndarray: An array of sorted detections.

    """
    # We sort the detection by decreasing confidence
    sort_indices = np.argsort(detections[:, 1])[::-1]
    return detections[sort_indices, :]
