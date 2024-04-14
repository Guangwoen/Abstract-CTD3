"""!

@brief The module contains K-Means algorithm and other related services.
@details Implementation based on paper @cite inproceedings::kmeans::1.

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2020
@copyright BSD-3-Clause

"""

import copy

import time
import numpy

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import pyclustering.core.kmeans_wrapper as wrapper

from pyclustering.core.wrapper import ccore_library
from pyclustering.core.metric_wrapper import metric_wrapper

from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster import cluster_visualizer

from pyclustering.utils.metric import distance_metric, type_metric
import utils

class kmeans:
    """!
    @brief Class implements K-Means clustering algorithm.
    @details K-Means clustering aims to partition n observations into k clusters in which each observation belongs to
              the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning
              of the data space into Voronoi cells.

    K-Means clustering results depend on initial centers. Algorithm K-Means++ can used for initialization of
    initial centers - see module 'pyclustering.cluster.center_initializer'.

    CCORE implementation (C/C++ part of the library) of the algorithm performs parallel processing to ensure maximum
    performance.

    Implementation based on the paper @cite inproceedings::kmeans::1.

    @image html kmeans_example_clustering.png "Fig. 1. K-Means clustering results. At the left - 'Simple03.data' sample, at the right - 'Lsun.data' sample."

    Example #1 - Clustering using K-Means++ for center initialization:
    @code
        from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
        from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
        from pyclustering.samples.definitions import FCPS_SAMPLES
        from pyclustering.utils import read_sample

        # Load list of points for cluster analysis.
        sample = read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS)

        # Prepare initial centers using K-Means++ method.
        initial_centers = kmeans_plusplus_initializer(sample, 2).initialize()

        # Create instance of K-Means algorithm with prepared centers.
        kmeans_instance = kmeans(sample, initial_centers)

        # Run cluster analysis and obtain results.
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        final_centers = kmeans_instance.get_centers()

        # Visualize obtained results
        kmeans_visualizer.show_clusters(sample, clusters, final_centers)
    @endcode

    Example #2 - Clustering using specific distance metric, for example, Manhattan distance:
    @code
        # prepare input data and initial centers for cluster analysis using K-Means

        # create metric that will be used for clustering
        manhattan_metric = distance_metric(type_metric.MANHATTAN)

        # create instance of K-Means using specific distance metric:
        kmeans_instance = kmeans(sample, initial_centers, metric=manhattan_metric)

        # run cluster analysis and obtain results
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
    @endcode

    @see center_initializer

    """

    def __init__(self, data, initial_centers, config, tolerance=0.001, ccore=True, **kwargs):
        """!
        @brief Constructor of clustering algorithm K-Means.
        @details Center initializer can be used for creating initial centers, for example, K-Means++ method.

        @param[in] data (array_like): Input data that is presented as array of points (objects), each point should be represented by array_like data structure.
        @param[in] initial_centers (array_like): Initial coordinates of centers of clusters that are represented by array_like data structure: [center1, center2, ...].
        @param[in] tolerance (double): Stop condition: if maximum value of change of centers of clusters is less than tolerance then algorithm stops processing.
        @param[in] ccore (bool): Defines should be CCORE library (C++ pyclustering library) used instead of Python code or not.
        @param[in] **kwargs: Arbitrary keyword arguments (available arguments: 'observer', 'metric', 'itermax').

        <b>Keyword Args:</b><br>
            - observer (kmeans_observer): Observer of the algorithm to collect information about clustering process on each iteration.
            - metric (distance_metric): Metric that is used for distance calculation between two points (by default euclidean square distance).
            - itermax (uint): Maximum number of iterations that is used for clustering process (by default: 200).

        @see center_initializer

        """
        self.config = config
        self.__pointer_data = numpy.array(data)
        self.__clusters = []
        self.__centers = numpy.array(initial_centers)
        self.__tolerance = tolerance
        self.__total_wce = 0.0

        self.__observer = kwargs.get('observer', None)
        self.__metric = copy.copy(kwargs.get('metric', distance_metric(type_metric.EUCLIDEAN_SQUARE)))
        self.__itermax = kwargs.get('itermax', 300)

        if self.__metric.get_type() != type_metric.USER_DEFINED:
            self.__metric.enable_numpy_usage()
        else:
            self.__metric.disable_numpy_usage()

        self.__ccore = ccore and self.__metric.get_type() != type_metric.USER_DEFINED
        if self.__ccore is True:
            self.__ccore = ccore_library.workable()

        self.__verify_arguments()

    def process(self):
        """!
        @brief Performs cluster analysis in line with rules of K-Means algorithm.

        @return (kmeans) Returns itself (K-Means instance).

        @see get_clusters()
        @see get_centers()

        """

        if len(self.__pointer_data[0]) != len(self.__centers[0]):
            raise ValueError("Dimension of the input data and dimension of the initial cluster centers must be equal.")

        if self.__ccore is True:
            self.__process_by_ccore()
        else:
            self.__process_by_python()

        return self

    def __process_by_ccore(self):
        """!
        @brief Performs cluster analysis using CCORE (C/C++ part of pyclustering library).

        """
        ccore_metric = metric_wrapper.create_instance(self.__metric)

        results = wrapper.kmeans(self.__pointer_data, self.__centers, self.__tolerance, self.__itermax,
                                 (self.__observer is not None), ccore_metric.get_pointer())

        self.__clusters = results[0]
        self.__centers = results[1]

        if self.__observer is not None:
            self.__observer.set_evolution_clusters(results[2])
            self.__observer.set_evolution_centers(results[3])

        self.__total_wce = results[4][0]

    def __process_by_python(self):
        """!
        @brief Performs cluster analysis using python code.

        """

        maximum_change = float('inf')
        iteration = 0

        if self.__observer is not None:
            initial_clusters = self.__update_clusters()
            self.__observer.notify(initial_clusters, self.__centers.tolist())

        while maximum_change > self.__tolerance and iteration < self.__itermax:

            self.__clusters = self.__update_clusters()
            updated_centers = self.__update_centers()  # changes should be calculated before assignment
            print("中心点个数", len(self.__clusters))


            if self.__observer is not None:
                self.__observer.notify(self.__clusters, updated_centers.tolist())

            maximum_change = self.__calculate_changes(updated_centers)

            self.__centers = updated_centers  # assign center after change calculation
            iteration += 1
        print("迭代次数： ", iteration)
        self.__calculate_total_wce()

    def predict(self, points):
        """!
        @brief Calculates the closest cluster to each point.

        @param[in] points (array_like): Points for which closest clusters are calculated.

        @return (list) List of closest clusters for each point. Each cluster is denoted by index. Return empty
                 collection if 'process()' method was not called.

        """

        nppoints = numpy.array(points)
        if len(self.__clusters) == 0:
            return []

        differences = numpy.zeros((len(nppoints), len(self.__centers)))
        for index_point in range(len(nppoints)):
            if self.__metric.get_type() != type_metric.USER_DEFINED:
                differences[index_point] = self.__metric(nppoints[index_point], self.__centers)
            else:
                differences[index_point] = [self.__metric(nppoints[index_point], center) for center in self.__centers]

        return numpy.argmin(differences, axis=1)

    def get_clusters(self):
        """!
        @brief Returns list of allocated clusters, each cluster contains indexes of objects in list of data.

        @see process()
        @see get_centers()

        """

        return self.__clusters

    def get_centers(self):
        """!
        @brief Returns list of centers of allocated clusters.

        @see process()
        @see get_clusters()

        """

        if isinstance(self.__centers, list):
            return self.__centers

        return self.__centers.tolist()

    def get_total_wce(self):
        """!
        @brief Returns sum of metric errors that depends on metric that was used for clustering (by default SSE - Sum of Squared Errors).
        @details Sum of metric errors is calculated using distance between point and its center:
                 \f[error=\sum_{i=0}^{N}distance(x_{i}-center(x_{i}))\f]

        @see process()
        @see get_clusters()

        """

        return self.__total_wce

    def get_cluster_encoding(self):
        """!
        @brief Returns clustering result representation type that indicate how clusters are encoded.

        @return (type_encoding) Clustering result representation.

        @see get_clusters()

        """

        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION


    def __update_clusters(self):
        """!
        @brief Calculate distance (in line with specified metric) to each point from the each cluster. Nearest points
                are captured by according clusters and as a result clusters are updated.

        @return (list) Updated clusters as list of clusters. Each cluster contains indexes of objects from data.

        """

        clusters = [[] for _ in range(len(self.__centers))]
        dataset_differences = self.__calculate_dataset_difference(len(clusters))
        optimum_indexes = numpy.argmin(dataset_differences, axis=0)
        for index_point in range(len(optimum_indexes)):
            index_cluster = optimum_indexes[index_point]
            clusters[index_cluster].append(index_point)

        clusters = [cluster for cluster in clusters if len(cluster) > 0]

        return clusters

    def __update_centers(self):
        """!
        @brief Calculate centers of clusters in line with contained objects.

        @return (numpy.array) Updated centers.

        """

        dimension = self.__pointer_data.shape[1]
        centers = numpy.zeros((len(self.__clusters), dimension))

        for index in range(len(self.__clusters)):
            cluster_points = self.__pointer_data[self.__clusters[index], :]
            centers[index] = cluster_points.mean(axis=0)
            centers[index] = utils.intervalize_state(centers[index], self.config)

        return numpy.array(centers)


    def __calculate_total_wce(self):
        """!
        @brief Calculate total within cluster errors that is depend on metric that was chosen for K-Means algorithm.

        """

        dataset_differences = self.__calculate_dataset_difference(len(self.__clusters))

        self.__total_wce = 0.0
        for index_cluster in range(len(self.__clusters)):
            for index_point in self.__clusters[index_cluster]:
                self.__total_wce += dataset_differences[index_cluster][index_point]

    def __calculate_dataset_difference(self, amount_clusters):
        """!
        @brief Calculate distance from each point to each cluster center.

        """
        dataset_differences = numpy.zeros((amount_clusters, len(self.__pointer_data)))
        for index_center in range(amount_clusters):
            if self.__metric.get_type() != type_metric.USER_DEFINED:
                dataset_differences[index_center] = self.__metric(self.__pointer_data, self.__centers[index_center])
            else:
                dataset_differences[index_center] = [self.__metric(point, self.__centers[index_center])
                                                     for point in self.__pointer_data]

        return dataset_differences

    def __calculate_changes(self, updated_centers):
        """!
        @brief Calculates changes estimation between previous and current iteration using centers for that purpose.

        @param[in] updated_centers (array_like): New cluster centers.

        @return (float) Maximum changes between centers.

        """
        if len(self.__centers) != len(updated_centers):
            maximum_change = float('inf')

        else:
            if self.__metric.get_type() != type_metric.USER_DEFINED:
                changes = self.__metric(self.__centers, updated_centers)
            else:
                changes = [self.__metric(center, updated_center) for center, updated_center in
                           zip(self.__centers, updated_centers)]

            maximum_change = numpy.max(changes)

        return maximum_change

    def __verify_arguments(self):
        """!
        @brief Verify input parameters for the algorithm and throw exception in case of incorrectness.

        """
        if len(self.__pointer_data) == 0:
            raise ValueError("Input data is empty (size: '%d')." % len(self.__pointer_data))

        if len(self.__centers) == 0:
            raise ValueError("Initial centers are empty (size: '%d')." % len(self.__pointer_data))

        if self.__tolerance < 0:
            raise ValueError("Tolerance (current value: '%d') should be greater or equal to 0." %
                             self.__tolerance)

        if self.__itermax < 0:
            raise ValueError("Maximum iterations (current value: '%d') should be greater or equal to 0." %
                             self.__tolerance)
