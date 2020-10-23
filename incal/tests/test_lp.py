import random
import numpy as np

from pywmi import Domain, sample, evaluate
from pywmi.smt_print import pretty_print

from incal import learn_incremental
from incal.lp_learner_milp import LpLearnerMilp
from incal.parameter_free_learner import LpSearchStrategy
from incal.violations.core import RandomViolationsStrategy


def get_domain(n, ranges=None, var_range=None):
    if ranges is None:
        ranges = [var_range or (None, None) for i in range(n)]
    return Domain.make([], ["x{}".format(i + 1) for i in range(n)], ranges)


def test_lp_milp_box():
    domain = get_domain(2, var_range=(0, 1))
    x, y = domain.get_symbols()
    formula = (x > 0.1) & (x < 0.3) & (y > 0.2) & (y < 0.4)

    random.seed(666)
    np.random.seed(666)
    data = sample.uniform(domain, 500)
    labels = evaluate(domain, formula, data)

    def initial(_options):
        return random.sample(_options, 20)

    learned_formula, final_k, final_h = learn_incremental(
        domain,
        data,
        labels,
        lambda k, h, ss: LpLearnerMilp(h, ss, False),
        initial,
        RandomViolationsStrategy(10),
        LpSearchStrategy(),
    )

    print(data)
    print(labels)
    print(pretty_print(learned_formula))

    assert np.all(labels == evaluate(domain, learned_formula, data))
