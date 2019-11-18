from pysmt.shortcuts import Real

from typing import Tuple, Optional

import numpy as np
from pysmt.fnode import FNode
from pywmi.domain import Density, Domain

from incal.observe.inc_logging import LoggingObserver
from incal.parameter_free_learner import learn_bottom_up


class Formula(Density):
    def __init__(self, domain, support):
        super().__init__(domain, support, Real(1))

    @classmethod
    def from_state(cls, state: dict):
        density = Density.from_state(state)
        return cls(density.domain, density.support)


def learn(
    domain: Domain,
    data: np.ndarray,
    labels: np.ndarray,
    learner_factory: callable,
    initial_strategy: callable,
    selection_strategy: object,
    initial_k: int,
    initial_h: int,
    weight_k: float,
    weight_h: float,
    log: Optional[str] = None,
) -> Tuple[FNode, int, int]:
    """
    Learn a formula that separates the positive and negative examples
    :return: A tuple containing 1. the learned formula, 2. the number of terms (or clauses) used,
    3. the number of hyperplanes used
    """

    # log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "results")
    # problem_name = hashlib.sha256(name).hexdigest()

    def learn_inc(_data, _labels, _i, _k, _h):
        learner = learner_factory(_k, _h, selection_strategy)
        initial_indices = initial_strategy(list(range(len(_data))))
        # log_file = os.path.join(log_dir, "{}_{}_{}.txt".format(problem_name, _k, _h))
        if log is not None:
            learner.add_observer(
                LoggingObserver(log, _k, _h, None, False, selection_strategy)
            )
        return learner.learn(domain, _data, _labels, initial_indices)

    ((_d, _l, formula), k, h) = learn_bottom_up(
        data, labels, learn_inc, weight_k, weight_h, initial_k, initial_h, None, None
    )
    return formula, k, h
