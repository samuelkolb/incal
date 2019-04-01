import os
import random

import numpy
from autodora.experiment import Experiment, Parameter, Result
from pysmt.fnode import FNode

from pywmi import evaluate
from pywmi.sample import uniform

from genratorsyn import Formula


class SyntheticExperiment(Experiment):
    file = Parameter(str, None, "The formula filename")
    learner = Parameter(str, None, "The learning algorithm")
    sample_size = Parameter(int, None, "The number of samples to learn from")
    learned_formula = Result(FNode, None, "The learned formula")
    log_file = Result(str, None, "The results log file")
    seed = Result(int, None, "The seed used to solve")

    def formula(self):
        return Formula.from_file(self["file"])

    def h(self):
        return len(self.formula().support.args())

    @staticmethod
    def np_to_dict(domain, samples, labels):
        results = []
        for s, l in zip(samples, labels):
            variables = {}
            for i, p in zip(domain.variables, range(len(domain.variables))):
                variables[i] = s[p].item()
            results.append((variables, l))
        return results

    def run_internal(self):
        from lplearing import learn_parameter_free

        formula = self.formula()
        samples = uniform(formula.domain, self["sample_size"])
        labels = evaluate(formula.domain, formula.support, samples)
        data = SyntheticExperiment.np_to_dict(formula.domain, samples, labels)
        seed = random.randint(0, 10000000)
        self["seed"] = seed
        random.seed(seed)
        numpy.random.seed(seed)
        log_file_name = "{}_{}_{}".format(self["file"], self["sample_size"], seed)
        if not os.path.exists(log_file_name):
            os.makedirs(log_file_name)
        self["log_file"] = log_file_name
        self["learned_formula"] = learn_parameter_free(formula.support, data, log_file_name, self["learner"])


SyntheticExperiment.enable_cli()
