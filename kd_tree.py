"""
CS5800 Summer 2024
Final Project
Author: Calvin Lo

This module implements a k-d tree for matching nearest neighbors.
The matches are stored in a cv2.DMatch object so it is compatible with OpenCV
functions.
"""
import cv2
import heapq

# constants
MAX_BINS = 25  # max bins to explore for BBF


class KD_Node:
    """
    Class -- KD_Node
        Represents a node in a k-d tree.

    Attributes:
        point: The k-dimensional point stored at this node.
        index: The index of the point in the original dataset.
        left_child: The left child of this node in the k-d tree.
        right_child: The right child of this node in the k-d tree.
    """
    def __init__(self, point, index, left=None, right=None):
        """
        Constructor -- __init__
        Initializes the KD_Node with a point, its index, and optional left
        and right children.

        Parameters:
            point: A k-dimensional point.
            index: The index of the point in the dataset.
            left: The left child node (optional).
            right: The right child node (optional).
        """
        self.point = point
        self.index = index
        self.left_child = left
        self.right_child = right


def build_tree(points, depth, k):
    """
    Function -- build_tree
        Recursively builds a k-d tree from a list of points.

    Parameters:
        points: A list of tuples, where each tuple contains an index and a
                k-dimensional point.
        depth: The current depth in the tree, used to determine the axis of
               splitting.
        k: The number of dimensions (k) of the points.

    Returns:
        KD_Node: The root node of the k-d tree.
    """
    if not points:
        return None

    # find the partitioning axis
    axis = depth % k

    # find the median
    points.sort(key=lambda x: x[1][axis])
    median = len(points) // 2

    # build left and right children
    return KD_Node(
        point=points[median][1],
        index=points[median][0],
        left=build_tree(points[:median], depth + 1, k),
        right=build_tree(points[median + 1:], depth + 1, k)
    )


def brute_force_nn2(points, query_point, k):
    """
    Function -- brute_force_nn2
        Finds the two nearest neighbors of a query point using a brute-force
        approach.

    Parameters:
        points: A list of tuples, where each tuple contains an index and a
                k-dimensional point.
        query_point: The k-dimensional query point.
        k: The number of dimensions (k) of the points.

    Returns:
        tuple: A tuple containing the two nearest neighbors and their squared
               distances. Each neighbor is represented as a tuple of the form
               (distance, point, index).
    """
    # Initialize variables
    closest_point = None
    second_closest_point = None
    min_dist = float('inf')
    second_min_dist = float('inf')
    closest_point_idx = None

    for point in points:
        dist = dist_squared(point[1], query_point, k)

        # Update the closest and second closest points
        if dist < min_dist:
            second_min_dist = min_dist
            second_closest_point = closest_point
            second_closest_point_idx = closest_point_idx
            min_dist = dist
            closest_point = point[1]
            closest_point_idx = point[0]
        elif dist < second_min_dist:
            second_min_dist = dist
            second_closest_point = point[1]
            second_closest_point_idx = point[0]

    # Return the closest and second closest points along with their distances
    return (min_dist, closest_point, closest_point_idx), \
        (second_min_dist, second_closest_point, second_closest_point_idx)


def dist_squared(vector1, vector2, k):
    """
    Function -- dist_squared
        Calculates the squared Euclidean distance between two k-dimensional
        points.

    Parameters:
        vector1: The first k-dimensional point.
        vector2: The second k-dimensional point.
        k: The number of dimensions (k) of the points.

    Returns:
        float: The squared Euclidean distance between vector1 and vector2.
    """
    sum = 0
    for i in range(k):
        sum += (vector1[i] - vector2[i]) ** 2
    return sum


def nearest_neighbor(node, query_point, k, depth=0):
    """
    Function -- nearest_neighbor
        Recursively finds the nearest neighbor of a query point in a k-d tree.

    Parameters:
        node: The current KD_Node in the k-d tree.
        query_point: The k-dimensional query point.
        k: The number of dimensions (k) of the points.
        depth: The current depth in the tree, used to determine the axis of
               splitting.

    Returns:
        tuple: A tuple containing the nearest neighbor, its squared distance,
               and its index in the original dataset.
    """

    # base case
    if node is None:
        return None, float('inf'), None

    # find parititioning dimension
    axis = depth % k

    # keep track of branches for backtracking
    next_branch = None
    opposite_branch = None
    if query_point[axis] < node.point[axis]:
        next_branch = node.left_child
        opposite_branch = node.right_child
    else:
        next_branch = node.right_child
        opposite_branch = node.left_child

    # recursively search the child node
    best, min_dist, index = \
        nearest_neighbor(next_branch, query_point, k, depth + 1)

    # check if distance at current node is less than min distance found so far
    current_dist = dist_squared(node.point, query_point, k)
    if best is None or current_dist < min_dist:
        best = node.point
        min_dist = current_dist
        index = node.index

    # check if there could be a nearer point in unexplored branch
    if (query_point[axis] - node.point[axis])**2 < min_dist:
        opposite_best, opposite_min_dist, opposite_index = \
            nearest_neighbor(opposite_branch, query_point, k, depth + 1)

        if opposite_min_dist < min_dist:
            best = opposite_best
            min_dist = opposite_min_dist
            index = opposite_index

    return best, min_dist, index


def two_nearest_neighbor(node, query_point, k, depth=0):
    """
    Function -- two_nearest_neighbor
        Recursively finds the two nearest neighbors of a query point in a k-d
        tree.

    Parameters:
        node: The current KD_Node in the k-d tree.
        query_point: The k-dimensional query point.
        k: The number of dimensions (k) of the points.
        depth: The current depth in the tree, used to determine the axis of
               splitting.

    Returns:
        list: A list containing the two nearest neighbors. Each neighbor is
              represented as a tuple of the form (distance, point, index).
    """
    # base case
    if node is None:
        return [(float('inf'), None, None), (float('inf'), None, None)]

    # get partition axis
    axis = depth % k

    next_branch = None
    opposite_branch = None
    if query_point[axis] < node.point[axis]:
        next_branch = node.left_child
        opposite_branch = node.right_child
    else:
        next_branch = node.right_child
        opposite_branch = node.left_child

    # check next branch
    best_neighbors = two_nearest_neighbor(next_branch,
                                          query_point,
                                          k, depth + 1)

    # update best guess based on current node
    current_dist = dist_squared(node.point, query_point, k)
    if current_dist < best_neighbors[0][0]:
        best_neighbors[1] = best_neighbors[0]
        best_neighbors[0] = (current_dist, node.point, node.index)
    elif current_dist < best_neighbors[1][0]:
        best_neighbors[1] = (current_dist, node.point, node.index)

    # check opposite branch
    if (query_point[axis] - node.point[axis])**2 < best_neighbors[1][0]:
        opposite_best_neighbors = \
            two_nearest_neighbor(opposite_branch,
                                 query_point,
                                 k,
                                 depth + 1)
        if opposite_best_neighbors[1][0] < best_neighbors[0][0]:
            best_neighbors[1] = best_neighbors[0]
            best_neighbors[0] = opposite_best_neighbors[1]
        elif opposite_best_neighbors[1][0] < best_neighbors[1][0]:
            best_neighbors[1] = opposite_best_neighbors[1]

        if opposite_best_neighbors[0][0] < best_neighbors[0][0]:
            best_neighbors[1] = best_neighbors[0]
            best_neighbors[0] = opposite_best_neighbors[0]
        elif opposite_best_neighbors[0][0] < best_neighbors[1][0]:
            best_neighbors[1] = opposite_best_neighbors[0]

    return best_neighbors


def best_bin_first_two(root, query_point, k, max_nodes=float('inf')):
    """
    Function -- best_bin_first_two
        Performs an approximate nearest neighbor search using the
        Best-Bin-First (BBF) algorithm, finding the two nearest neighbors.

    Parameters:
        root: The root node of the k-d tree.
        query_point: The k-dimensional query point.
        k: The number of dimensions (k) of the points.
        max_nodes: The maximum number of nodes to explore.

    Returns:
        list: A list containing the two nearest neighbors. Each neighbor is
              represented as a tuple of the form (distance, point, index).
    """
    # base case
    best_neighbors = [(float('inf'), None, None), (float('inf'), None, None)]

    if root is None:
        return best_neighbors

    # Priority queue to hold nodes to explore
    # Elements are tuples of (distance to the splitting plane, node, depth)
    pq = []
    heapq.heappush(pq, (0, 0, root, 0))
    insert_order = 0
    counter = 0
    # dist, node, depth = heapq.heappop(pq)
    while pq and max_nodes > counter:
        counter += 1
        # Pop the node with the smallest distance to the partition plane
        _, _, node, depth = heapq.heappop(pq)

        axis = depth % k

        # Calculate squared distance between the query point and the current
        # node's point
        current_dist = dist_squared(node.point, query_point, k)
        if current_dist < best_neighbors[0][0]:
            best_neighbors[1] = best_neighbors[0]
            best_neighbors[0] = (current_dist, node.point, node.index)
        elif current_dist < best_neighbors[1][0]:
            best_neighbors[1] = (current_dist, node.point, node.index)    

        # Determine which branch to explore first
        if query_point[axis] < node.point[axis]:
            next_branch = node.left_child
            opposite_branch = node.right_child
        else:
            next_branch = node.right_child
            opposite_branch = node.left_child

        # Add the next branch to the priority queue
        if next_branch is not None:
            insert_order += 1
            heapq.heappush(pq, (0, insert_order, next_branch, depth + 1))

        # Calculate distance to the splitting plane
        plane_dist = (query_point[axis] - node.point[axis])**2

        # If the distance to the splitting plane is less than the minimum
        # distance found, add the opposite branch to the priority queue
        if plane_dist < best_neighbors[1][0] and opposite_branch is not None:
            insert_order += 1
            heapq.heappush(pq, (plane_dist,
                                insert_order,
                                opposite_branch,
                                depth + 1))

    return best_neighbors


def get_descriptors(image_path):
    """
    Function -- get_descriptors
        Extracts SIFT descriptors from an image.

    Parameters:
        image_path: The path to the image file.

    Returns:
        tuple: A tuple containing keypoints and descriptors extracted from the
               image.
    """
    # Create a SIFT detector object
    sift = cv2.SIFT_create()

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def custom_match_features(descriptors, query_descriptors, k, method):
    """
    Function -- custom_match_features
        Matches query descriptors to a set of descriptors using a specified
        nearest neighbor search method. The results are stored in a cv2.DMatch
        object for compatibility with OpenCV.

    Parameters:
        descriptors: A set of SIFT descriptors.
        query_descriptors: A set of query descriptors to match against.
        k: The number of dimensions (k) of the descriptors.
        method: The method of matching ('nn' for nearest neighbor, 'bf' for
                brute force, or 'bbf' for best-bin-first).

    Returns:
        list: A list of tuples, each containing two cv2.DMatch objects
              representing the best and second-best matches.
    """
    # generate indices for each keypoint
    indexed_points = \
        [(i, point) for i, point in enumerate(descriptors.tolist())]
    
    # build k-d tree
    node = build_tree(indexed_points, 0, k)
    matches = []

    for i, descriptor in enumerate(query_descriptors):
        match method:
            case 'nn':
                best_neighbors = two_nearest_neighbor(node, descriptor, k)
            case 'bf':
                best_neighbors = brute_force_nn2(descriptors, descriptor, k)
            case 'bbf':
                best_neighbors = \
                    best_bin_first_two(node, descriptor, k, MAX_BINS)

        m = cv2.DMatch()
        n = cv2.DMatch()
        m.queryIdx = i
        m.trainIdx = best_neighbors[0][2]
        m.distance = best_neighbors[0][0]
        n.queryIdx = i
        n.trainIdx = best_neighbors[1][2]
        n.distance = best_neighbors[1][0]
        matches.append((m, n))

    return matches


def ratio_test(matches, ratio=0.80):
    """
    Function -- ratio_test
        Filters matches based on the ratio test. The ratio test compares the
        distance of the best match to the distance of the second-best match,
        and only keeps matches where the best match is sufficiently closer.

    Parameters:
        matches: A list of tuples, each containing two cv2.DMatch objects
                 representing the best and second-best matches.
        ratio: The ratio threshold. A smaller ratio results in more strict
               filtering.

    Returns:
        list: A list of cv2.DMatch objects that passed the ratio test.
    """
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches
