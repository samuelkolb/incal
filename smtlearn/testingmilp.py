from milp import  rockingthemilp
from milp_as_in_paper import papermilp
import pywmi
from smt_print import pretty_print

from lplearing import polutionreduction, cuben,sample_half_half,learn_parameter_free
from pysmt.shortcuts import Real


model=polutionreduction()
sample=sample_half_half(model,100)

theory_one, x, y=learn_parameter_free(model, sample, 10)
theory_two=rockingthemilp(model.domain, sample, 3)
theory_three=papermilp(model.domain, sample, 3)


sample_count=10000000

tpr1 = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
                                      sample_count).compute_probability(theory_one)
tnr1 = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
                                      sample_count).compute_probability(~theory_one)


tpr2 = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
                                      sample_count).compute_probability(theory_two)
tnr2 = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
                                      sample_count).compute_probability(~theory_two)

tpr3 = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
                                      sample_count).compute_probability(theory_three)
tnr3 = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
                                      sample_count).compute_probability(~theory_three)