import random

from pysmt.shortcuts import Real

from typing import Tuple, Optional, Callable, List

import numpy as np
from pysmt.fnode import FNode
from pywmi.domain import Density, Domain

from incal.incremental_learner import IncrementalLearner
from incal.observe.inc_logging import LoggingObserver
from incal.parameter_free_learner import (
    learn_bottom_up,
    SearchStrategy,
    DoubleSearchStrategy,
    LpSearchStrategy,
)
from incal.violations.core import SelectionStrategy, RandomViolationsStrategy


class Formula(Density):
    def __init__(self, domain, support):
        super().__init__(domain, support, Real(1))

    @classmethod
    def from_state(cls, state: dict):
        density = Density.from_state(state)
        return cls(density.domain, density.support)


Indices = List[int]


def learn_incremental(
    domain: Domain,
    data: np.ndarray,
    labels: np.ndarray,
    learner_factory: Callable[[int, int, SelectionStrategy], IncrementalLearner],
    initial_strategy: Callable[[Indices], Indices],
    selection_strategy: SelectionStrategy,
    search_strategy: SearchStrategy,
    log: Optional[str] = None,
) -> Tuple[FNode, int, int]:
    """
    Learn a formula that separates the positive and negative examples
    :return: A tuple containing 1. the learned formula, 2. the number of terms (or clauses) used,
    3. the number of hyperplanes used
    """

    def learn_inc(_data, _labels, _i, _k, _h):
        iteration_learner = learner_factory(_k, _h, selection_strategy)
        initial_indices = initial_strategy(list(range(len(_data))))
        # log_file = os.path.join(log_dir, "{}_{}_{}.txt".format(problem_name, _k, _h))
        if log is not None:
            iteration_learner.add_observer(
                LoggingObserver(log, _k, _h, None, False, selection_strategy)
            )
        return iteration_learner.learn(domain, _data, _labels, initial_indices)

    ((_d, _l, formula), k, h) = learn_bottom_up(
        data, labels, learn_inc, search_strategy
    )
    return formula, k, h


def incal(
    domain: Domain,
    data: np.ndarray,
    labels: np.ndarray,
) -> Tuple[FNode, int, int]:
    from incal.k_cnf_smt_learner import KCnfSmtLearner

    return learn_incremental(
        domain,
        data,
        labels,
        lambda k, h, ss: KCnfSmtLearner(k, h, ss, ""),
        lambda ind: random.sample(ind, min(data.shape[0], 20)),
        RandomViolationsStrategy(10),
        DoubleSearchStrategy(1.4, 1),
    )


def incalp(
    domain: Domain,
    data: np.ndarray,
    labels: np.ndarray,
) -> Tuple[FNode, int]:
    from incal.lp_learner_milp import LpLearnerMilp

    formula, k, h = learn_incremental(
        domain,
        data,
        labels,
        lambda _k, _h, ss: LpLearnerMilp(_h, ss),
        lambda ind: random.sample(ind, min(data.shape[0], 20)),
        RandomViolationsStrategy(10),
        LpSearchStrategy(),
    )
    return formula, k
