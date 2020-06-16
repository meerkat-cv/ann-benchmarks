from __future__ import absolute_import
import numpy as np


def knn_threshold(data, count, epsilon):
    return data[count - 1] + epsilon


def epsilon_threshold(data, count, epsilon):
    return data[count - 1] * (1 + epsilon)


def get_recall_values(dataset_distances, run_distances, count, threshold,
                      epsilon=1e-3):
    """
    Recall based on distances to neighbors, allow distance error of epslion
    """
    recalls = np.zeros(len(run_distances))
    for i in range(len(run_distances)):
        t = threshold(dataset_distances[i], count, epsilon)
        actual = 0
        for d in run_distances[i][:count]:
            if d <= t:
                actual += 1
        recalls[i] = actual
    return (np.mean(recalls) / float(count),
            np.std(recalls) / float(count),
            recalls)


def get_recall_from_sets(dataset_neighbors, run_neighbors, k, threshold):
    """
    Recall base o set on intersection of elements from ANN with the KNN ground-truth
    """
    recalls = np.zeros(len(run_neighbors))
    for i in range(len(run_neighbors)):
        I = np.intersect1d(run_neighbors[i],dataset_neighbors[i][:k])
        recalls[i] = len(I)

    return (np.mean(recalls) / float(k),
            np.std(recalls) / float(k),
            recalls)


def knnDist(dataset_distances, run_distances, count, metrics, epsilon=1e-3):
    """
    Computes the knn-recall based on distances to neighbors, allow distance error of epslion.
    """
    group='knnDist'
    if group not in metrics:
        print('Computing knn metrics')
        knn_metrics = metrics.create_group(group)
        mean, std, recalls = get_recall_values(dataset_distances,
                                               run_distances, count,
                                               knn_threshold, epsilon)

        knn_metrics.attrs['mean'] = mean
        knn_metrics.attrs['std'] = std
        knn_metrics['recalls'] = recalls
    else:
        print("Found cached result")
    return metrics[group]


def knnSet(dataset_neighbors, run_neighbors, k, metrics):
    """
    Computes the recall of a query based on the interection os Run_neighbors and the dataset neighbors.
    """
    group='knnSet'
    if group not in metrics:
        print('Computing knn metrics')
        knn_metrics = metrics.create_group(group)
        mean, std, recalls = get_recall_from_sets(dataset_neighbors,
                                                  run_neighbors, k,
                                                  knn_threshold)
        knn_metrics.attrs['mean'] = mean
        knn_metrics.attrs['std'] = std
        knn_metrics['recalls'] = recalls
    else:
        print("Found cached result")
    return metrics[group]


def epsilon(dataset_distances, run_distances, count, metrics, epsilon=0.01):
    s = 'eps' + str(epsilon)
    if s not in metrics:
        print('Computing epsilon metrics')
        epsilon_metrics = metrics.create_group(s)
        mean, std, recalls = get_recall_values(dataset_distances,
                                               run_distances, count,
                                               epsilon_threshold, epsilon)
        epsilon_metrics.attrs['mean'] = mean
        epsilon_metrics.attrs['std'] = std
        epsilon_metrics['recalls'] = recalls
    else:
        print("Found cached result")
    return metrics[s]


def get_rank_values( query_labels, run_labels ):
    rank = []
    for q, neighbors in zip(query_labels, run_labels):
        pos_arr = np.where(neighbors == q)
        if len(pos_arr[0]) == 0:
            r = float('inf')
        else:
            r = pos_arr[0][0]

        rank.append(r)
    return np.asarray(rank)


def get_accuracy_rank(rank_values, rank_n = 1):
    filtered_rank = np.where(rank_values < rank_n)
    accuracy = len(filtered_rank[0])/len(rank_values)
    return accuracy


def accuracy(query_labels, run_labels, metrics):
    """
    Accuracy if the label is within the first rank-n neighbors
    - count is the K for knn computed.
    """

    s = 'accuracy'
    if s not in metrics:
        print('Computing rank metrics')
        rank_metrics = metrics.create_group(s)
        rank_values = get_rank_values(query_labels, run_labels)

        R1 = get_accuracy_rank(rank_values, 1)
        R10 = get_accuracy_rank(rank_values, 10)

        rank_metrics.attrs['R@1'] = R1
        rank_metrics.attrs['R@10'] = R10
        rank_metrics['ranks'] = rank_values

    else:
        print("Found cached result")
    return metrics[s]


def rel(dataset_distances, run_distances, metrics):
    if 'rel' not in metrics.attrs:
        print('Computing rel metrics')
        total_closest_distance = 0.0
        total_candidate_distance = 0.0
        for true_distances, found_distances in zip(dataset_distances,
                                                   run_distances):
            for rdist, cdist in zip(true_distances, found_distances):
                total_closest_distance += rdist
                total_candidate_distance += cdist
        if total_closest_distance < 0.01:
            metrics.attrs['rel'] = float("inf")
        else:
            metrics.attrs['rel'] = total_candidate_distance / \
                total_closest_distance
    else:
        print("Found cached result")
    return metrics.attrs['rel']


def queries_per_second(queries, attrs):
    return 1.0 / attrs["best_search_time"]

def search_seconds(queries, attrs):
    #print(attrs['best_search_time'])
    return attrs["best_search_time"]


def index_size(queries, attrs):
    # TODO(erikbern): should replace this with peak memory usage or something
    return attrs.get("index_size", 0)


def build_time(queries, attrs):
    return attrs["build_time"]


def candidates(queries, attrs):
    return attrs["candidates"]


def dist_computations(queries, attrs):
    return attrs.get("dist_comps", 0) / (attrs['run_count'] * len(queries))


all_metrics = {
    "knn-dist": {
        "description": "Distance Recall",
        "function": lambda true_distances, run_distances, metrics, run_attrs, *args: knnDist(true_distances, run_distances, run_attrs["count"], metrics).attrs['mean'],  # noqa
        "worst": float("-inf"),
        "lim": [0.0, 1.03]
    },
    "knn-set": {
        "description": "Set Recall",
        "function": lambda _, __, metrics, run_attrs, ___, ____, true_neighbors, run_neighbors: knnSet(true_neighbors, run_neighbors, run_attrs["count"], metrics).attrs['mean'],  # noqa
        "worst": float("-inf"),
    },
#    "epsilon": {
#        "description": "Epsilon 0.01 Recall",
#        "function": lambda true_distances, run_distances, metrics, run_attrs, *args: epsilon(true_distances, run_distances, run_attrs["count"], metrics).attrs['mean'],  # noqa
#        "worst": float("-inf")
#    },
#    "largeepsilon": {
#        "description": "Epsilon 0.1 Recall",
#        "function": lambda true_distances, run_distances, metrics, run_attrs, *args: epsilon(true_distances, run_distances, run_attrs["count"], metrics, 0.1).attrs['mean'],  # noqa
#        "worst": float("-inf")
#    },
    "accuracy-R@1": {
        "description": "Accuracy-R@1",
        "function": lambda _ , __ , metrics, run_attrs , query_labels, run_labels, *args: accuracy(query_labels, run_labels, metrics).attrs['R@1'],  # noqa
        "worst": float("-inf")
    },
    "accuracy-R@10": {
        "description": "Accuracy-R@10",
        "function": lambda _ , __ , metrics, run_attrs , query_labels, run_labels, *args: accuracy(query_labels, run_labels, metrics).attrs['R@10'],  # noqa
        "worst": float("-inf")
    },
#    "rel": {
#        "description": "Relative Error",
#        "function": lambda true_distances, run_distances, metrics, run_attrs, *args: rel(true_distances, run_distances, metrics),  # noqa
#        "worst": float("inf")
#    },
    "qps": {
        "description": "Queries per second (1/s)",
        "function": lambda true_distances, run_distances, metrics, run_attrs, *args: queries_per_second(true_distances, run_attrs),  # noqa
        "worst": float("-inf")
    },
    "queryTime": {
        "description": "Best query search time (s)",
        "function": lambda true_distances, run_distances, metrics, run_attrs, *args: search_seconds(true_distances, run_attrs),  # noqa
        "worst": float("inf")
    },
#    "distcomps": {
#        "description": "Distance computations",
#        "function": lambda true_distances, run_distances,  metrics, run_attrs, *args: dist_computations(true_distances, run_attrs), # noqa
#        "worst": float("inf")
#    },
    "build": {
        "description": "Build time (s)",
        "function": lambda true_distances, run_distances, metrics, run_attrs, *args: build_time(true_distances, run_attrs), # noqa
        "worst": float("inf")
    },
#    "candidates": {
#        "description": "Candidates generated",
#        "function": lambda true_distances, run_distances, metrics, run_attrs, *args: candidates(true_distances, run_attrs), # noqa
#        "worst": float("inf")
#    },
    "indexsize": {
        "description": "Index size (kB)",
        "function": lambda true_distances, run_distances, metrics, run_attrs, *args: index_size(true_distances, run_attrs),  # noqa
        "worst": float("inf")
    },
#    "queriessize": {
#        "description": "Index size (kB)/Queries per second (s)",
#        "function": lambda true_distances, run_distances, metrics, run_attrs, *args: index_size(true_distances, run_attrs) / queries_per_second(true_distances, run_attrs), # noqa
#        "worst": float("inf")
#    }
}
