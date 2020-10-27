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


def test_lp_milp_box_from_selected():
    domain = get_domain(2, var_range=(None, None))
    x, y = domain.get_symbols()
    formula = (x >= 0.3) & (x <= 0.8) & (y >= 0.2) & (y <= 0.9)

    data = np.array(
        [
            [0.3, 0.2],
            [0.3, 0.9],
            [0.8, 0.2],
            [0.8, 0.9],
            [0.25, 0.2],
            [0.25, 0.9],
            [0.85, 0.2],
            [0.85, 0.9],
            [0.2, 0.15],
            [0.2, 0.95],
            [0.8, 0.15],
            [0.8, 0.95],
        ]
    )
    labels = evaluate(domain, formula, data)

    def initial(_options):
        return random.sample(_options, 5)

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
    print(learned_formula)
    print(evaluate(domain, learned_formula, data))

    assert np.all(labels == evaluate(domain, learned_formula, data))


def test_lp_milp_box_from_spreadsheet():
    domain = get_domain(2, var_range=(None, None))
    x, y = domain.get_symbols()
    formula = (x >= 0.3) & (x <= 0.8) & (y >= 0.2) & (y <= 0.9)
    data = np.array(
        [
            [0.3, 0.2],
            [0.3, 0.9],
            [0.8, 0.2],
            [0.8, 0.9],
            [0.8181, 0.65598],
            [0.13633, 0.7631],
            [0.30709, 0.51138],
            [0.15993, 0.18114],
            [0.53445, 0.34425],
            [0.72953, 0.48604],
            [0.32356, 0.8551],
            [0.9494, 0.38718],
            [0.79876, 0.1992],
            [0.89698, 0.23736],
            [0.06101, 0.35284],
            [0.68164, 0.90693],
            [0.36606, 0.21165],
            [0.90486, 0.66886],
            [0.51002, 0.21007],
            [0.70245, 0.83331],
        ]
    )
    labels = evaluate(domain, formula, data)

    def initial(_options):
        return random.sample(_options, 5)

    learned_formula, final_k, final_h = learn_incremental(
        domain,
        data,
        labels,
        lambda k, h, ss: LpLearnerMilp(h, ss, False),
        initial,
        RandomViolationsStrategy(10),
        LpSearchStrategy(init_h=4, max_h=4),
    )

    print(data)
    print(labels)
    print(learned_formula)
    print(evaluate(domain, learned_formula, data))

    assert np.all(labels == evaluate(domain, learned_formula, data))
