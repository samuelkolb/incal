import heapq

import time
import logging

from .learner import NoFormulaFound

logger = logging.getLogger(__name__)


class ParameterFrontier(object):
    def __init__(self, cost_function):
        self.pq = []
        self.tried = set()
        self.cost_function = cost_function

    def push(self, k, h):
        if (k, h) not in self.tried:
            heapq.heappush(self.pq, (self.cost_function(k, h), k, h))
            self.tried.add((k, h))

    def pop(self):
        c, k, h = heapq.heappop(self.pq)
        return k, h


class SearchStrategy:
    def produce_initial(self):
        raise NotImplementedError()

    def produce_next(self, k, h):
        raise NotImplementedError()

    def cost(self, k, h):
        raise NotImplementedError()


class DoubleSearchStrategy(SearchStrategy):
    def __init__(self, w_k=1, w_h=1, init_k=1, init_h=0, max_k=None, max_h=None):
        self.w_k = w_k
        self.w_h = w_h
        self.init_k = init_k
        self.init_h = init_h
        self.max_k = max_k
        self.max_h = max_h

    def produce_initial(self):
        return self.init_k, self.init_h

    def produce_next(self, k, h):
        result = []
        if self.max_k is None or k + 1 <= self.max_k:
            result.append((k + 1, h))
        if self.max_h is None or h + 1 <= self.max_h:
            result.append((k, h + 1))
        return result

    def cost(self, k, h):
        return self.w_k * k + self.w_h * h


class LpSearchStrategy(SearchStrategy):
    def __init__(self, init_h=1, max_h=None):
        self.init_h = init_h
        self.max_h = max_h

    def produce_initial(self):
        return self.init_h, self.init_h

    def produce_next(self, k, h):
        return [] if (self.max_h is not None and h >= self.max_h) else [(k + 1, h + 1)]

    def cost(self, k, h):
        return h


def learn_bottom_up(data, labels, learn_f, search_strategy: SearchStrategy):
    """
    Learns a CNF(k, h) SMT formula phi using the learner encapsulated in init_learner such that
    C(k, h) = w_k * k + w_h * h is minimal.
    :param data: List of tuples of assignments and labels
    :param labels: Array of labels
    :param learn_f: Function called with data, k and h: learn_f(data, k, h)
    :param search_strategy: The search strategy
    :return: A tuple containing: 1) the CNF(k, h) formula phi with minimal complexity C(k, h); 2) k; and 3) h
    """
    solution = k = h = None
    frontier = ParameterFrontier(search_strategy.cost)
    frontier.push(*search_strategy.produce_initial())
    i = 0
    while solution is None:
        i += 1
        try:
            k, h = frontier.pop()
        except IndexError:
            return (data, labels, None), None, None

        logger.debug("Attempting to solve with k={} and h={}".format(k, h))
        start = time.time()
        try:
            solution = learn_f(data, labels, i, k, h)
            logger.debug("Found solution after {:.2f}s".format(time.time() - start))
        except NoFormulaFound as e:
            data = e.data
            labels = e.labels
        for next_node in search_strategy.produce_next(k, h):
            frontier.push(*next_node)
    return solution, k, h
