from __future__ import print_function

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import TimeoutError
from glob import glob

from autodora.observe import ProgressObserver
from autodora.runner import CommandLineRunner, PrintObserver
from autodora.sql_storage import SqliteStorage
from autodora.trajectory import product

from scipy.spatial import ConvexHull

from pebble import ProcessPool
from pysmt.shortcuts import *

import pywmi
from genratorsyn import generate_synthetic_formula, Formula
from inc_logging import LoggingObserver
from incremental_learner import IncrementalObserver, \
    MaxViolationsStrategy
from lp import LPLearner
from milp import LPLearnermilp
from milp_as_in_paper import papermilp
# from k_dnf_logic_learner import KDNFLogicLearner, GreedyLogicDNFLearner, GreedyMaxRuleLearner
# from k_dnf_smt_learner import KDnfSmtLearner
# from k_dnf_greedy_learner import GreedyMilpRuleLearner
from parameter_free_learner import learn_bottom_up
from parse import smt_to_nested, nested_to_smt
from pywmi import evaluate, Domain, Density
from reducedmilp import smallmilp
from pywmi.sample import uniform
from smt_check import SmtChecker
from smt_print import pretty_print


# from virtual_data import OneClassStrategy
#from milp import rockingthemilp
from syn_experiment import SyntheticExperiment


def evaluate_assignment(problem, assignment):
    # substitution = {problem.domain.get_symbol(v): val for v, val in assignment.items()}
    # return problem.theory.substitute(substitution).simplify().is_true()
    return SmtChecker(assignment).check(problem.theory)


def evaluate_assignment2(problem, assignment):
    # substitution = {problem.domain.get_symbol(v): val for v, val in assignment.items()}
    # return problem.theory.substitute(substitution).simplify().is_true()
    return SmtChecker(assignment).check(problem)

def normilisation_rightsideto1(theory):
    inequalitys = []
    if len(theory.get_atoms()) == 1:
        if theory.arg(1).constant_value() > 0:
            constant = theory.arg(1)
            leftside = theory.arg(0)
            le = Real(constant.constant_value() / constant.constant_value())

            inequalitys.append(GE(le, Plus(
                [leftside.arg(j).arg(0).constant_value() / constant.constant_value() * leftside.arg(j).arg(1) for j in
                 range(len(leftside.args()))])))
        elif theory.arg(1).constant_value() < 0:
            constant = theory.arg(1)
            leftside = theory.arg(0)
            le = Real(constant.constant_value() / constant.constant_value())
            inequalitys.append(LE(le, Plus(
                [leftside.arg(j).arg(0).constant_value() / constant.constant_value() * leftside.arg(j).arg(1) for j in
                 range(len(leftside.args()))])))
        else:
            return theory

    else:
        for i in range(len(theory.get_atoms())):
            if theory.arg(i).arg(1).constant_value() > 0:
                constant = theory.arg(i).arg(1)
                leftside = theory.arg(i).arg(0)

                le = Real(constant.constant_value() / constant.constant_value())

                inequalitys.append(GE(le, Plus(
                    [leftside.arg(j).arg(0).constant_value() / constant.constant_value() * leftside.arg(j).arg(1) for j
                     in range(len(leftside.args()))])))
            elif theory.arg(i).arg(1).constant_value() < 0:
                constant = theory.arg(i).arg(1)
                leftside = theory.arg(i).arg(0)

                le = Real(constant.constant_value() / constant.constant_value())

                inequalitys.append(LE(le, Plus(
                    [leftside.arg(j).arg(0).constant_value() / constant.constant_value() * leftside.arg(j).arg(1) for j
                     in range(len(leftside.args()))])))
            else:
                constant = theory.arg(i).arg(1)
                leftside = theory.arg(i).arg(0)

                le = Real(constant.constant_value())

                inequalitys.append(GE(le, Plus([leftside.arg(j).arg(0).constant_value() * leftside.arg(j).arg(1) for j in
                                                range(len(leftside.args()))])))

    normalisedtheory = And(inequalitys)
    return normalisedtheory

def sample(problem, n, seed=None):
    """
    Sample n instances for the given problem
    :param problem: The problem to sample from
    :param n: The number of samples
    :param seed: An optional seed
    :return: A list containing n samples and their label (True iff the sample satisfies the theory and False otherwise)
    """
    # TODO Other distributions
    samples = []
    for i in range(n):
        instance = dict()
        for v in problem.domain.variables:
            if problem.domain.var_types[v] == REAL:
                lb, ub = problem.domain.var_domains[v]
                instance[v] = random.uniform(lb, ub)
            elif problem.domain.var_types[v] == BOOL:
                instance[v] = True if random.random() < 0.5 else False
            else:
                raise RuntimeError("Unknown variable type {} for variable {}", problem.domain.var_types[v], v)
        samples.append((instance, evaluate_assignment(problem, instance)))

    return samples


def sample_half_half(problem, n, seed=None):
    """
    Sample n instances for the given problem
    :param problem: The problem to sample from
    :param n: The number of samples
    :param seed: An optional seed
    :return: A list containing n samples and their label (True iff the sample satisfies the theory and False otherwise)
    """
    # TODO Other distributions
    samplespositive = []
    samplesnegative = []
    while len(samplespositive) < n / 2 or len(samplesnegative) < n / 2:
        instance = dict()
        for v in problem.domain.variables:
            if problem.domain.var_types[v] == REAL:
                lb, ub = problem.domain.var_domains[v]
                instance[v] = random.uniform(lb, ub)
            else:
                raise RuntimeError("Unknown variable type {} for variable {}", problem.domain.var_types[v], v)
        # print(evaluate_assignment(problem, instance))

        if evaluate_assignment(problem, instance) and len(samplespositive) < n / 2:
            samplespositive.append((instance, evaluate_assignment(problem, instance)))
        elif len(samplesnegative) < n / 2:
            samplesnegative.append((instance, evaluate_assignment(problem, instance)))

    samples = samplesnegative + samplespositive
    random.shuffle(samples)
    return samples


class TrackingObserver(IncrementalObserver):

    def observe_initial(self, initial_indices):
        self.initials = [i for i in initial_indices]

    def observe_iteration(self, theory, new_active_indices, solving_time, selection_time):
        for i in new_active_indices:
            self.initials.append(i)

    def __init__(self, initial_indices):
        self.initials = [i for i in initial_indices]


class ParameterObserver:

    def __init__(self):
        self.seednumber = 0
        self.dimensions = 0
        self.samplesize = 0
        self.time = 0
        self.numberofconstrains = 0
        self.numberofconstrainsmodel = 0
        self.overallruns = []
        self.tpr = 0
        self.tnr = 0

    def updating(self):
        self.overallruns.append(
            [self.seednumber, self.dimensions, self.samplesize, self.time, self.numberofconstrains,
             self.numberofconstrainsmodel, self.tpr, self.tnr])

def loadintermedautresult(dic,v):
    lh=len(os.listdir(dic))-1
    print(lh)
    with open(dic+v+str(lh)+".csv") as f:
        data = [json.loads(line) for line in f]
    last = data[-1]
    return nested_to_smt(last["theory"])



def get_dt_weights(m, data):
    import dt_selection
    dt_weights = [min(d.values()) for d in dt_selection.get_distances(m.domain, data)]
    return dt_weights


def learn_parameter_free(problem, data, log_dir_name, method="smt"):
    #feat_x, feat_y = problem.domain.real_vars[:2]# needed for plotting observer

    #o = TrackingObserver(random.sample(list(range(len(data))), 20))

    w = get_dt_weights(problem, data)
    p = zip(range(len(data)), [w[i] for i in range(len(data))])
    p = sorted(p, key=lambda t: t[1])
    d = [t[0] for t in p[0:20]]
    o = TrackingObserver(d)


    def learn_inc(_data, i, _k, _h):

        w = get_dt_weights(problem, data)

        # learner = LPLearner(_h, RandomViolationsStrategy(10))
        # learner = LPLearner(_h, WeightedRandomViolationsStrategy(10,w))
        #out symetries
        #learner = (_h, MaxViolationsStrategy(1, w))# ,symmetries="n")
        if method=="smt":
            learner=LPLearner(_h, MaxViolationsStrategy(1, w))
            #learner = LPLearner(_h, MaxViolationsStrategy(1, w) ,symmetries="n")
        else:
            learner = LPLearnermilp(_h, MaxViolationsStrategy(1, w))

        #change temp to len(data)
        log_file = os.path.join(log_dir_name, "{}.csv".format(_h))


        #img_name = "{}_{}_{}_{}_{}_{}_{}".format(learner.name, len(problem.domain.variables), i, _k, _h, len(data),
         #                                        seed)
        #dir_nameP = "../output/{}/{}/{}/".format(problem.name, len(data), seed, len(data))
        #dir_nameP = "../output/{}/{}/{}/{}/".format("syn", problem.half_space_count, len(data), seed, len(data))
        #learner.add_observer(plotting.PlottingObserver(problem.domain, data, dir_nameP, img_name, feat_x, feat_y))

        learner.add_observer(LoggingObserver(log_file, verbose=False))

        learner.add_observer(o)

        if len(o.initials) > 20:
            initial_indices = random.sample(o.initials, 20)

        else:
            initial_indices = o.initials

        learned_theory = learner.learn(problem.domain, data, initial_indices)

        print("Learned theory:\n{}".format(pretty_print(learned_theory)))
        return learned_theory

    return learn_bottom_up(data, learn_inc, 1000000, 1)#,init_h=len(x.get_atoms()))


def learndouble(problem,data,seed):
    w = get_dt_weights(problem, data)
    p = zip(range(len(data)), [w[i] for i in range(len(data))])
    p = sorted(p, key=lambda t: t[1])
    d = [t[0] for t in p[0:20]] #[-20:]
    print("1",d)
    da=[data[i] for i in d]
    x,y,z=learn_parameter_free(problem,da,seed)
    print(z)

    return learn_parameter_free(problem,data,seed,z)


def nptodic(generated):
    samples=[]
    for s,l in zip(generated.samples, generated.labels):

        variables = {}
        for i ,p in zip(generated.formula.domain.variables ,range(len(generated.formula.domain.variables))):
            variables[i]=s[p].item()
        samples.append((variables,l))
    return samples


# with open("/Users/Elias/Documents/GitHub/smtlearn/output/incal milp/simplexn/theories_learned/D:4S:500.csv") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         theory_one=nested_to_smt(row[0])

# t=simplexn(4)
# sample_count=1000000
# tprl=[]
# tnrl=[]
# start = time.time()
# for i in range(50):
#     tpr = pywmi.RejectionEngine(t.domain, t.theory, Real(1.0),
#                                                       sample_count).compute_probability(theory_one)
#     tnr = pywmi.RejectionEngine(t.domain, ~t.theory, Real(1.0),
#                                                       sample_count).compute_probability(~theory_one)
#     tprl.append(tpr)
#     tnrl.append(tnr)
#
# print(time.time()-start)
# print(np.std(tprl))
# print(np.std(tnrl))
# print(np.mean(tprl))
# print(np.mean(tnrl))



def testing(nbseeds, nbdimensions, nbsamplesize, modelinput,incal="smt"):
    parameter = ParameterObserver()
    sample_count = 1000000

    for i in nbdimensions:

        model = modelinput(i)
        dir_name_for_equaltions = "../output/{}/{}/{}/".format(incal,model.name, "theories_learned")
        if not os.path.exists(dir_name_for_equaltions):
            os.makedirs(dir_name_for_equaltions)

        for j in nbsamplesize:
            parameter.seednumber = 0
            equationsfound = []
            seeds = random.sample(range(10000000), nbseeds)
            dir_name_for_samplesize = "../output/{}/{}/{}".format(incal,model.name, j)
            if not os.path.exists(dir_name_for_samplesize):
                os.makedirs(dir_name_for_samplesize)

            for k in seeds:

                dir_name_for_seed = "../output/{}/{}/{}/{}".format(incal,model.name, j, k)
                if not os.path.exists(dir_name_for_seed):
                    os.makedirs(dir_name_for_seed)
                print(time.asctime(), i, j, k)
                parameter.seednumber = parameter.seednumber + 1
                parameter.dimensions = i
                parameter.samplesize = j

                parameter.numberofconstrains = len(model.theory.get_atoms())

                data = sample_half_half(model, j, k)

                start = time.time()
                if incal == "smt":
                    with ProcessPool() as pool:
                        future = pool.schedule(learn_parameter_free, args=[model, data, k,incal], timeout=90*60)

                    try:
                        result = future.result()
                        learned_theory = result[0]
                        km = result[1]
                        numberofconstrains = result[2]
                        parameter.time = time.time() - start
                    except TimeoutError:
                        dir_nameO = "../output/{}/{}/{}/{}/".format(incal,model.name, len(data), k)
                        dir_run = "observer{}:{}:".format(len(data), len(model.domain.variables))
                        parameter.time = False
                        learned_theory = loadintermedautresult(dir_nameO, dir_run)
                        numberofconstrains = len(learned_theory.get_atoms())
                        print("Function took longer than seconds")
                elif incal == "milp":
                    with ProcessPool() as pool:
                        future = pool.schedule(learn_parameter_free, args=[model, data, k,incal], timeout=90*60)

                    try:
                        result = future.result()
                        learned_theory = result[0]
                        km = result[1]
                        numberofconstrains = result[2]
                        parameter.time = time.time() - start
                    except TimeoutError:
                        dir_nameO = "../output/{}/{}/{}/{}/".format(incal,model.name, len(data), k)
                        dir_run = "observer{}:{}:".format(len(data), len(model.domain.variables))
                        parameter.time = False
                        learned_theory = loadintermedautresult(dir_nameO, dir_run)
                        numberofconstrains = len(learned_theory.get_atoms())
                        print("Function took longer than seconds")
                elif incal=="smallmilp":
                    learned_theory, numberofconstrains, timelimit = smallmilp(model.domain, data, 14)
                     #learned_theory, numberofconstrains = rockingthemilp(model.domain, data, 14)
                    if timelimit == True:
                        parameter.time = 90*60

                    else:
                        parameter.time = time.time() - start
                elif incal=="bigmilp":
                    # learned_theory, km, numberofconstrains = learn_parameter_free(model, data, k)
                    learned_theory, numberofconstrains, timelimit = papermilp(model.domain, data, 14)
                    # learned_theory, numberofconstrains = rockingthemilp(model.domain, data, 14)
                    if timelimit == True:
                        parameter.time = 90*60

                    else:
                        parameter.time = time.time() - start

                parameter.numberofconstrainsmodel = numberofconstrains
                # parameter.accucy=accucy(theroy=model,theorylearned=theory,data1=adata)
                parameter.tpr = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
                                                      sample_count).compute_probability(learned_theory)
                parameter.tnr = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
                                                      sample_count).compute_probability(~learned_theory)

                parameter.updating()
                equationsfound.append([smt_to_nested(learned_theory)])

            dir_name_for_equaltions = "../output/{}/{}/{}/D:{}S:{}.csv".format(incal,model.name, "theories_learned", i, j)
            with open(dir_name_for_equaltions, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(equationsfound)

    dir_name_results = "../output/{}/{}/{}{}{}.csv".format(incal,model.name, "Results", nbdimensions, nbsamplesize)
    with open(dir_name_results, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(parameter.overallruns)

    return parameter.overallruns

#testing(1,[2],[50],cuben,"smallmilp")

def testingpractical(nbseeds, nbsamplesize, modelinput,incal="smt"):
    parameter = ParameterObserver()
    sample_count = 1000000
    model = modelinput()
    dir_name_for_equaltions = "../output/{}/{}/{}/".format(incal,model.name, "theories_learned")
    if not os.path.exists(dir_name_for_equaltions):
        os.makedirs(dir_name_for_equaltions)

    for j in nbsamplesize:
        parameter.seednumber = 0
        equationsfound = []
        seeds = random.sample(range(10000000), nbseeds)
        dir_name_for_samplesize = "../output/{}/{}/{}".format(incal,model.name, j)
        if not os.path.exists(dir_name_for_samplesize):
            os.makedirs(dir_name_for_samplesize)
        for k in seeds:

            dir_name_for_seed = "../output/{}/{}/{}/{}".format(incal,model.name, j, k)
            if not os.path.exists(dir_name_for_seed):
                os.makedirs(dir_name_for_seed)
            print(time.asctime(), j, k)
            parameter.seednumber = parameter.seednumber + 1
            parameter.samplesize = j

            parameter.numberofconstrains = len(model.theory.get_atoms())
            data = sample_half_half(model, j, k)#change 100 to j

            start = time.time()
            if incal == "smt":
                with ProcessPool() as pool:
                    future = pool.schedule(learn_parameter_free, args=[model, data, k, incal], timeout=90 * 60)

                try:
                    result = future.result()
                    learned_theory = result[0]
                    km = result[1]
                    numberofconstrains = result[2]
                    parameter.time = time.time() - start
                except TimeoutError:
                    dir_nameO = "../output/{}/{}/{}/{}/".format(incal, model.name, len(data), k)
                    dir_run = "observer{}:{}:".format(len(data), len(model.domain.variables))
                    parameter.time = False
                    learned_theory = loadintermedautresult(dir_nameO, dir_run)
                    numberofconstrains = len(learned_theory.get_atoms())
                    print("Function took longer than seconds")
            elif incal == "milp":
                with ProcessPool() as pool:
                    future = pool.schedule(learn_parameter_free, args=[model, data, k, incal], timeout=90 * 60)

                try:
                    result = future.result()
                    learned_theory = result[0]
                    km = result[1]
                    numberofconstrains = result[2]
                    parameter.time = time.time() - start
                except TimeoutError:
                    dir_nameO = "../output/{}/{}/{}/{}/".format(incal, model.name, len(data), k)
                    dir_run = "observer{}:{}:".format(len(data), len(model.domain.variables))
                    parameter.time = False
                    learned_theory = loadintermedautresult(dir_nameO, dir_run)
                    numberofconstrains = len(learned_theory.get_atoms())
                    print("Function took longer than seconds")
            elif incal == "smallmilp":
                learned_theory, numberofconstrains, timelimit = smallmilp(model.domain, data, 14)
                # learned_theory, numberofconstrains = rockingthemilp(model.domain, data, 14)
                if timelimit == True:
                    parameter.time = 90 * 60

                else:
                    parameter.time = time.time() - start
            elif incal == "bigmilp":
                # learned_theory, km, numberofconstrains = learn_parameter_free(model, data, k)
                learned_theory, numberofconstrains, timelimit = papermilp(model.domain, data, 14)
                # learned_theory, numberofconstrains = rockingthemilp(model.domain, data, 14)
                if timelimit == True:
                    parameter.time = 90 * 60

                else:
                    parameter.time = time.time() - start


            parameter.numberofconstrainsmodel = numberofconstrains


            parameter.tpr = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
                                                  sample_count).compute_probability(learned_theory)
            parameter.tnr = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
                                                  sample_count).compute_probability(~learned_theory)

            # parameter.accucy=accucy(theroy=model,theorylearned=theory,size=100,seed=k+1)
            parameter.updating()
            equationsfound.append([smt_to_nested(learned_theory)])

        dir_name_for_equaltions = "../output/{}/{}/{}/S:{}.csv".format(incal,model.name, "theories_learned", j)

        with open(dir_name_for_equaltions, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(equationsfound)

    dir_name = "../output/{}/{}/{}{}{}.csv".format(incal,model.name, "Results", model.name, nbsamplesize)
    with open(dir_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(parameter.overallruns)

    return parameter.overallruns
#testingpractical(1,[50],police,"milp")
#from genratorsyn import generate_synthetic_dataset
#x=generate_synthetic_dataset("synthetic",0,3,"cnf",2,1,2,500,70)

#d=nptodic(x)

def testingsyn(nbseeds, numberofvariables, nbsamplesize, numberofhalfspaces, incal="smt"):
    #parameter = ParameterObserver()
    sample_count = 1000000
    from genratorsyn import generate_synthetic_dataset
    for a in numberofhalfspaces:
        parameter = ParameterObserver()
        for i in numberofvariables:

            dir_name_for_equaltions = "../output/{}/{}/{}/{}/".format(incal,"syn",a, "theories_learned")
            if not os.path.exists(dir_name_for_equaltions):
                os.makedirs(dir_name_for_equaltions)

            for j in nbsamplesize:
                parameter.seednumber = 0
                equationsfound = []
                seeds = random.sample(range(10000000), nbseeds)
                dir_name_for_samplesize = "../output/{}/{}/{}/{}".format(incal,"syn",a, j)
                if not os.path.exists(dir_name_for_samplesize):
                    os.makedirs(dir_name_for_samplesize)

                for k in seeds:

                    dir_name_for_seed = "../output/{}/{}/{}/{}/{}".format(incal,"syn",a, j, k)
                    if not os.path.exists(dir_name_for_seed):
                        os.makedirs(dir_name_for_seed)
                    print(time.asctime(), i, j, k)
                    parameter.seednumber = parameter.seednumber + 1
                    parameter.dimensions = i
                    parameter.samplesize = j

                    parameter.numberofconstrains = a
                    try:
                        synmodel = generate_synthetic_dataset("synthetic", 0, i, "cnf", a, 1, a, j, 70)
                    except RuntimeError:
                        continue

                    data=nptodic(synmodel)
                    pretty_print(synmodel.formula.support)

                    start = time.time()
                    if incal == "smt":
                        with ProcessPool() as pool:
                            future = pool.schedule(learn_parameter_free, args=[synmodel.formula, data, k,incal,True], timeout=90*60)

                        try:
                            result = future.result()
                            learned_theory = result[0]
                            km = result[1]
                            numberofconstrains = result[2]
                            parameter.time = time.time() - start
                        except TimeoutError:
                            dir_nameO = "../output/{}/{}/{}/{}/{}/".format(incal,"syn",a, len(data), k)
                            dir_run = "observer{}:{}:".format(len(data), i)
                            parameter.time = 90*60
                            learned_theory = loadintermedautresult(dir_nameO, dir_run)
                            numberofconstrains = len(learned_theory.get_atoms())
                            print("Function took longer than seconds")
                    elif incal == "milp":
                        with ProcessPool() as pool:
                            future = pool.schedule(learn_parameter_free, args=[synmodel.formula, data, k,incal,True], timeout=90*60)

                        try:
                            result = future.result()
                            learned_theory = result[0]
                            km = result[1]
                            numberofconstrains = result[2]
                            parameter.time = time.time() - start
                        except TimeoutError:
                            dir_nameO = "../output/{}/{}/{}/{}/{}/".format(incal,"syn",a, len(data), k)
                            dir_run = "observer{}:{}:".format(len(data), i)
                            parameter.time = 90*60
                            learned_theory = loadintermedautresult(dir_nameO, dir_run)
                            numberofconstrains = len(learned_theory.get_atoms())
                            print("Function took longer than seconds")
                    elif incal =="smallmilp":
                        learned_theory, numberofconstrains, timelimit = smallmilp(synmodel.formula.domain, data, 14)
                        # learned_theory, numberofconstrains = rockingthemilp(model.domain, data, 14)
                        if timelimit == True:
                            parameter.time = 90 * 60

                        else:
                            parameter.time = time.time() - start
                    elif incal =="bigmilp":
                        # learned_theory, km, numberofconstrains = learn_parameter_free(model, data, k)
                        learned_theory, numberofconstrains, timelimit = papermilp(synmodel.formula.domain, data, 14)
                        # learned_theory, numberofconstrains = rockingthemilp(model.domain, data, 14)
                        if timelimit == True:
                            parameter.time = 90*60

                        else:
                            parameter.time = time.time() - start

                    parameter.numberofconstrainsmodel = numberofconstrains
                    # parameter.accucy=accucy(theroy=model,theorylearned=theory,data1=adata)
                    parameter.tpr = pywmi.RejectionEngine(synmodel.formula.domain, synmodel.formula.support, Real(1.0),
                                                          sample_count).compute_probability(learned_theory)
                    parameter.tnr = pywmi.RejectionEngine(synmodel.formula.domain, ~synmodel.formula.support, Real(1.0),
                                                          sample_count).compute_probability(~learned_theory)

                    parameter.updating()
                    equationsfound.append([smt_to_nested(learned_theory)])

                dir_name_for_equaltions = "../output/{}/{}/{}/{}/D:{}S:{}.csv".format(incal,"syn",a, "theories_learned", i, j)
                with open(dir_name_for_equaltions, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(equationsfound)

        dir_name_results = "../output/{}/{}/{}/{}{}{}.csv".format(incal,"syn",a, "Results", numberofvariables, nbsamplesize)
        with open(dir_name_results, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(parameter.overallruns)

    return parameter.overallruns

#seeds,vars,samplesize,halfspaces


def test_synthetic(v, h, method):
    sample_sizes = [20, 30, 40, 50, 100, 200, 300, 400, 500]
    testingsyn(10, [v], sample_sizes, [h], method)


def gen_polytope(domain, n, h, ratio, test_sample_count):
    points = uniform(domain, n)
    convex_hull = ConvexHull(points)
    equations = convex_hull.equations
    inequalities = []
    for i in range(len(equations)):
        inequality = Real(equations[i, -1].item())
        for j in range(len(domain.real_vars)):
            inequality += domain.get_symbol(domain.real_vars[j]) * Real(equations[i, j].item())
        inequalities.append(inequality <= Real(0))
    test_samples = uniform(domain, test_sample_count)
    convex_hull_formula = And(*random.sample(inequalities, h))
    labels = evaluate(domain, convex_hull_formula, test_samples)
    if 1 - ratio <= sum(labels) / len(labels) <= ratio:
        print(sum(labels) / len(labels))
        return convex_hull_formula
    else:
        print("Failed")
        raise RuntimeError()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="task")

    synthetic = subparsers.add_parser("syn")
    synthetic.add_argument("dir")
    synthetic.add_argument("method")

    generator = subparsers.add_parser("gen")
    generator.add_argument("dir")
    generator.add_argument("v", type=int)
    generator.add_argument("h", type=int)
    generator.add_argument("s", type=int)
    generator.add_argument("-n", default=1, type=int)

    args = parser.parse_args()

    if args.task == "syn":
        files = list(glob(os.path.join(args.dir, "*.json")))
        file_dict = {"file": files}
        sample_size_dict = {"sample_size": [20, 30, 40, 50, 100, 200, 300, 400, 500]}
        learner_dict = {"learner": [args.method]}
        settings = product(file_dict, sample_size_dict, learner_dict)
        trajectory = SyntheticExperiment.explore("syn", settings)
        storage = SqliteStorage()
        dispatcher = ProgressObserver()
        dispatcher.add_observer(PrintObserver())
        CommandLineRunner(trajectory, storage, processes=1, timeout=1000, observer=dispatcher).run()

    elif args.task == "gen":
        v, s, h, n = args.v, args.s, args.h, args.n
        sample_count = 10000
        results = 0
        domain = Domain.make([], ["x{}".format(i) for i in range(v)], real_bounds=(0, 1))
        while results < args.n:
            seed = random.randint(0, 1000000000)
            try:
                random.seed(seed)
                formula = gen_polytope(domain, s, h, 0.7, sample_count)
                Formula(domain, formula).to_file(os.path.join(args.dir, "syn_{}_{}_{}_{}.json".format(v, h, s, seed)))
                results += 1
            except RuntimeError:
                pass


    #print(testing(1,[2],[50],cuben,"bigmilp"))
    #print(testingpractical(1,[50],polutionreduction,"milp"))


    # testing(10,[4],[400,500],cuben,"bigmilp")
    # testing(10,[4],[400,500],simplexn,"bigmilp")
    # testing(10,[4],[20,30,40,50,100,200,300,400,500],cuben,"bigmilp")
    # testing(10,[4],[20,30,40,50,100,200,300,400,500],simplexn,"bigmilp")
    #
    # testingpractical(10,[400,500,1000,2000],police,"bigmilp")
    # testingpractical(10,[400,500,1000,2000],polutionreduction(),"bigmilp")
    #
    # testing(10,[5,6],[200,300,400,500],cuben,"smt")
    # testing(10,[5,6],[200,300,400,500],simplexn,"smt")

    #
    # testing(10,[5,6],[200,300,400,500],cuben,"milp")
    # testing(10,[5,6],[200,300,400,500],simplexn,"milp")
    #
    #
    # testing(10,[4],[400,500],cuben,"smallmilp")
    # testing(10,[4],[400,500],simplexn,"smallmilp")

    # testing(10,[4],[20,30,40,50,100,200,300,400,500],cuben,"smallmilp")
    # testing(10,[4],[20,30,40,50,100,200,300,400,500],simplexn,"smallmilp")
    # testingpractical(10,[400,500,1000,2000],police,"smallmilp")
    # testingpractical(10,[400,500,1000,2000],polutionreduction(),"smallmilp")

    # testingsyn(10,[3],[20,30,40,50,100,200,300,400,500],[2,4,8,16,32,84],"smt")
    #testingsyn(10,[3],[20,30,40,50,100,200,300,400,500],[2,4,8,16,32,84],"milp")

    #testingsyn(10,[2,4,8,16,32,82],[20,30,40,50,100,200,300,400,500],[3],"smt")
    #testingsyn(10,[2,4,8,16,32,82],[20,30,40,50,100,200,300,400,500],[3],"milp")
    # random.seed(65)


if __name__ == '__main__':
    main()
