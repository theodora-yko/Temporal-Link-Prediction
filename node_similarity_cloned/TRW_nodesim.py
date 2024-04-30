import numpy as np
import warnings
from collections import defaultdict, deque
from scipy import stats
from scipy.special import softmax
from stellargraph import StellarGraph
from stellargraph.core.schema import GraphSchema
from stellargraph.core.utils import is_real_iterable
from stellargraph.core.experimental import experimental
from stellargraph.random import random_state


class GraphWalk(object):
    """
    Base class for exploring graphs.
    """

    def __init__(self, graph, graph_schema=None, seed=None):
        self.graph = graph

        # Initialize the random state
        self._check_seed(seed)

        self._random_state, self._np_random_state = random_state(seed)

        # We require a StellarGraph for this
        if not isinstance(graph, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph.")

        if not graph_schema:
            self.graph_schema = self.graph.create_graph_schema()
        else:
            self.graph_schema = graph_schema

        if type(self.graph_schema) is not GraphSchema:
            self._raise_error(
                "The parameter graph_schema should be either None or of type GraphSchema."
            )

    def get_adjacency_types(self):
        # Allow additional info for heterogeneous graphs.
        adj = getattr(self, "adj_types", None)
        if not adj:
            # Create a dict of adjacency lists per edge type, for faster neighbour sampling from graph in SampledHeteroBFS:
            self.adj_types = adj = self.graph._adjacency_types(self.graph_schema)
        return adj

    def _check_seed(self, seed):
        if seed is not None:
            if type(seed) != int:
                self._raise_error(
                    "The random number generator seed value, seed, should be integer type or None."
                )
            if seed < 0:
                self._raise_error(
                    "The random number generator seed value, seed, should be non-negative integer or None."
                )

    def _get_random_state(self, seed):
        """
        Args:
            seed: The optional seed value for a given run.

        Returns:
            The random state as determined by the seed.
        """
        if seed is None:
            # Restore the random state
            return self._random_state
        # seed the random number generator
        rs, _ = random_state(seed)
        return rs

    def neighbors(self, node):
        if not self.graph.has_node(node):
            self._raise_error("node {} not in graph".format(node))
        return self.graph.neighbors(node)

    def run(self, *args, **kwargs):
        """
        To be overridden by subclasses. It is the main entry point for performing random walks on the given
        graph.

        It should return the sequences of nodes in each random walk.
        """
        raise NotImplementedError

    def _raise_error(self, msg):
        raise ValueError("({}) {}".format(type(self).__name__, msg))

    def _check_common_parameters(self, nodes, n, length, seed):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: <list> A list of root node ids from which to commence the random walks.
            n: <int> Number of walks per node id.
            length: <int> Maximum length of each walk.
            seed: <int> Random number generator seed.
        """
        self._check_nodes(nodes)
        self._check_repetitions(n)
        self._check_length(length)
        self._check_seed(seed)

    def _check_nodes(self, nodes):
        if nodes is None:
            self._raise_error("A list of root node IDs was not provided.")
        if not is_real_iterable(nodes):
            self._raise_error("Nodes parameter should be an iterable of node IDs.")
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            warnings.warn(
                "No root node IDs given. An empty list will be returned as a result.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _check_repetitions(self, n):
        if type(n) != int:
            self._raise_error(
                "The number of walks per root node, n, should be integer type."
            )
        if n <= 0:
            self._raise_error(
                "The number of walks per root node, n, should be a positive integer."
            )

    def _check_length(self, length):
        if type(length) != int:
            self._raise_error("The walk length, length, should be integer type.")
        if length <= 0:
            # Technically, length 0 should be okay, but by consensus is invalid.
            self._raise_error("The walk length, length, should be a positive integer.")

    # For neighbourhood sampling
    def _check_sizes(self, n_size):
        err_msg = "The neighbourhood size must be a list of non-negative integers."
        if not isinstance(n_size, list):
            self._raise_error(err_msg)
        if len(n_size) == 0:
            # Technically, length 0 should be okay, but by consensus it is invalid.
            self._raise_error("The neighbourhood size list should not be empty.")
        for d in n_size:
            if type(d) != int or d < 0:
                self._raise_error(err_msg)

def naive_weighted_choices(rs, weights):
    """
    Select an index at random, weighted by the iterator `weights` of
    arbitrary (non-negative) floats. That is, `x` will be returned
    with probability `weights[x]/sum(weights)`.

    For doing a single sample with arbitrary weights, this is much (5x
    or more) faster than numpy.random.choice, because the latter
    requires a lot of preprocessing (normalized probabilties), and
    does a lot of conversions/checks/preprocessing internally.
    """

    # divide the interval [0, sum(weights)) into len(weights)
    # subintervals [x_i, x_{i+1}), where the width x_{i+1} - x_i ==
    # weights[i]
    subinterval_ends = []
    running_total = 0
    for w in weights:
        if w < 0:
            raise ValueError("Detected negative weight: {}".format(w))
        running_total += w
        subinterval_ends.append(running_total)

    # pick a place in the overall interval
    x = rs.random() * running_total

    # find the subinterval that contains the place, by looking for the
    # first subinterval where the end is (strictly) after it
    for idx, end in enumerate(subinterval_ends):
        if x < end:
            break

    return idx

class NS_TemporalRandomWalk(GraphWalk):
    """
    Performs temporal random walks on the given graph. The graph should contain numerical edge
    weights that correspond to the time at which the edge was created. Exact units are not relevant
    for the algorithm, only the relative differences (e.g. seconds, days, etc).
    """
    def run(
        self,
        num_cw,
        cw_size,
        max_walk_length=80,
        initial_edge_bias=None,
        walk_bias=None,
        p_walk_success_threshold=0.01,
        seed=None,
    ):
        """
        Perform a time respecting random walk starting from randomly selected temporal edges.

        Args:
            num_cw (int): Total number of context windows to generate. For comparable
                results to most other random walks, this should be a multiple of the number
                of nodes in the graph.
            cw_size (int): Size of context window. Also used as the minimum walk length,
                since a walk must generate at least 1 context window for it to be useful.
            max_walk_length (int): Maximum length of each random walk. Should be greater
                than or equal to the context window size.
            initial_edge_bias (str, optional): Distribution to use when choosing a random
                initial temporal edge to start from. Available options are:

                * None (default) - The initial edge is picked from a uniform distribution.
                * "exponential" - Heavily biased towards more recent edges.

            walk_bias (str, optional): Distribution to use when choosing a random
                neighbour to walk through. Available options are:

                * None (default) - Neighbours are picked from a uniform distribution.
                * "exponential" - Exponentially decaying probability, resulting in a bias towards shorter time gaps.

            p_walk_success_threshold (float): Lower bound for the proportion of successful
                (i.e. longer than minimum length) walks. If the 95% percentile of the
                estimated proportion is less than the provided threshold, a RuntimeError
                will be raised. The default value of 0.01 means an error is raised if less than 1%
                of the attempted random walks are successful. This parameter exists to catch any
                potential situation where too many unsuccessful walks can cause an infinite or very
                slow loop.
            seed (int, optional): Random number generator seed; default is None.

        Returns:
            List of lists of node ids for each of the random walks.

        """
        if cw_size < 2:
            raise ValueError(
                f"cw_size: context window size should be greater than 1, found {cw_size}"
            )
        if max_walk_length < cw_size:
            raise ValueError(
                f"max_walk_length: maximum walk length should not be less than the context window size, found {max_walk_length}"
            )

        np_rs = self._np_random_state if seed is None else np.random.RandomState(seed)
        walks = []
        num_cw_curr = 0

        edges, times = self.graph.edges(include_edge_weight=True)
        edge_biases = self._temporal_biases(
            times, None, bias_type=initial_edge_bias, is_forward=False,
        )

        successes = 0
        failures = 0

        def not_progressing_enough():
            # Estimate the probability p of a walk being long enough; the 95% percentile is used to
            # be more stable with respect to randomness. This uses Beta(1, 1) as the prior, since
            # it's uniform on p
            posterior = stats.beta.ppf(0.95, 1 + successes, 1 + failures)
            return posterior < p_walk_success_threshold

        # loop runs until we have enough context windows in total
        while num_cw_curr < num_cw:
            first_edge_index = self._sample(len(edges), edge_biases, np_rs)
            src, dst = edges[first_edge_index]
            t = times[first_edge_index]

            remaining_length = num_cw - num_cw_curr + cw_size - 1

            walk = self._walk(
                src, dst, t, min(max_walk_length, remaining_length), walk_bias, np_rs
            )
            if len(walk) >= cw_size:
                walks.append(walk)
                num_cw_curr += len(walk) - cw_size + 1
                successes += 1
            else:
                failures += 1
                if not_progressing_enough():
                    raise RuntimeError(
                        f"Discarded {failures} walks out of {failures + successes}. "
                        "Too many temporal walks are being discarded for being too short. "
                        f"Consider using a smaller context window size (currently cw_size={cw_size})."
                    )

        return walks


    def _sample(self, n, biases, np_rs):
        if biases is not None:
            assert len(biases) == n
            return naive_weighted_choices(np_rs, biases)
        else:
            return np_rs.choice(n)

    def _exp_biases(self, times, t_0, decay):
        # t_0 assumed to be smaller than all time values
        return softmax(t_0 - np.array(times) if decay else np.array(times) - t_0)

    def _temporal_biases(self, times, time, bias_type, is_forward):
        if bias_type is None:
            # default to uniform random sampling
            return None

        # time is None indicates we should obtain the minimum available time for t_0
        t_0 = time if time is not None else min(times)

        if bias_type == "exponential":
            # exponential decay bias needs to be reversed if looking backwards in time
            return self._exp_biases(times, t_0, decay=is_forward)
        else:
            raise ValueError("Unsupported bias type")

    def _step(self, node, time, bias_type, np_rs):
        """
        Perform 1 temporal step from a node. Returns None if a dead-end is reached.

        """
        print(node)
        neighbours = [
            (neighbour, t)
            for neighbour, t in self.graph.neighbors(node, include_edge_weight=True)
            if t > time
        ]
        
        def compute_jc(u_neighbours, v):
            v_neighbours = set(StellarGraph.neighbor_arrays(self.graph, v))
            union_size = len(u_neighbours.union(v_neighbours))
            if union_size == 0:
                return 0
            return len(u_neighbours.intersection(v_neighbours)) / union_size
        node_degrees = self.graph.node_degrees()

        if neighbours:
            times = [t for _, t in neighbours]
            biases = None
            node_degree = node_degrees[node]
            u_neighbours = set(StellarGraph.neighbor_arrays(self.graph, node))
            for ngh in neighbours: #G.neighbors(node):
                print(ngh)
                pval=compute_jc(u_neighbours, ngh) + 1.0/node_degree
                biases.append(pval)
                 
            # biases = self._temporal_biases(times, time, bias_type, is_forward=True)
            chosen_neighbour_index = self._sample(len(neighbours), biases, np_rs)
            next_node, next_time = neighbours[chosen_neighbour_index]
            return next_node, next_time
        else:
            return None

    def _walk(self, src, dst, t, length, bias_type, np_rs):
        walk = [src, dst]
        node, time = dst, t
        for _ in range(length - 2):
            print(node)
            result = self._step(node, time=time, bias_type=bias_type, np_rs=np_rs)

            if result is not None:
                node, time = result
                walk.append(node)
            else:
                break

        return walk
