

from gurobipy import *

#import lplearing
from problem import Domain, Problem

from pysmt.shortcuts import Real, LE, Plus, Times, Symbol,And, GE
from pysmt.typing import REAL
from smt_print import pretty_print
from lp_incremental_learner import IncrementalLearner

from z3.z3types import Z3Exception



class LPLearnermilp(IncrementalLearner):
    def __init__(self, half_space_count, selection_strategy, allow_negations=True, symmetries=""):
        IncrementalLearner.__init__(self, "cnf_smt", selection_strategy)
        self.half_space_count = half_space_count



    def learn_partial(self, solver, domain, inputdata, new_active_indices):
        m = Model("milp")
        m.setParam( 'OutputFlag', False )

        #constants
        n_c=self.half_space_count
        n_v=len(domain.real_vars) #number of varaibles #j
        bigm=10000000
        ep=1
        wmax=1000
        cmax=1000
        c_0=1

        data=inputdata.copy()
        data=[data[i] for i in new_active_indices]


        n_e = len(data)
        v_j_k = [[row[v] for v in domain.real_vars] for row, _ in data]


        #variables


        c_i = [m.addVar(vtype=GRB.CONTINUOUS, lb=-cmax, ub=cmax,name="c_i({i})".format(i=i))for i in range(n_c)]

        cb_i = [m.addVar(vtype=GRB.BINARY,name="cb_i({i})".format(i=i))for i in range(n_c)]

        w_i_j=[[m.addVar(vtype=GRB.CONTINUOUS, lb=-wmax, ub=wmax,name="w_i_j({i},{j})".format(i=i,j=j))for j in range(n_v)] for i in range(n_c)]

        wb_i_j=[[m.addVar(vtype=GRB.BINARY,name="wb_i_j({i},{j})".format(i=i,j=j))for j in range(n_v)] for i in range(n_c)]#for j in range(n_v)] for i in range(n_c)]

        s_k_i=[[m.addVar(vtype=GRB.BINARY,name="s_k_i({k},{i})".format(k=k,i=i))for i in range(n_c)] for k in range(n_e)]



        #1 sum(wij*xkj<=ci) A positives
        for k in range(n_e):
            if data[k][1]:
                for i in range(n_c): # on per postive exxample and number of constarints
                    m.addConstr(sum(w_i_j[i][j]*v_j_k[k][j] for j in range(n_v)) <= c_i[i], name="c1({k},{i})".format(k=k,i=i))
        #2 sum(wij*sjk>m*ski-M+ci+e) A negatives
        for k in range(n_e):
            if not data[k][1]:
                for i in range(n_c): # on per postive exxample and number of constarints
                    m.addConstr(sum(w_i_j[i][j]*v_j_k[k][j] for j in range(n_v)) >= bigm*s_k_i[k][i]- bigm+ c_i[i]+ep, name="c2({k},{i})".format(k=k,i=i))
        #3 sum(ski>=1) A negaives
        for k in range(n_e):# on per negative example
            if not data[k][1]:
                m.addConstr(sum(s_k_i[k][i]for i in range(n_c)) >= 1, name="c3({k})".format(k=k))

        #5wij<=cmax*wbij
        for i in range(n_c):
            for j in range(n_v):
                m.addConstr(w_i_j[i][j]<=wmax* wb_i_j[i][j],name="c5({i},{j})".format(i=i,j=j))
        #6wij>=-camxwbij
        for i in range(n_c):
            for j in range(n_v):
                m.addConstr(w_i_j[i][j]>=-wmax* wb_i_j[i][j],name="c6({i},{j})".format(i=i,j=j))
        #7ci<=cmaxcbi
        for i in range(n_c):
            m.addConstr(c_i[i]<=cmax*cb_i[i],name="c7({i})".format(i=i))
        #8ci>=-cmaxcbi
        for i in range(n_c):
            m.addConstr(c_i[i]>=-cmax*cb_i[i],name="c8({i})".format(i=i))
        #4 sum wbij>=cbi
        for i in range(n_c):
            m.addConstr(sum(wb_i_j[i][j] for j in range(n_v))>=cb_i[i],name="c4({i})".format(i=i))


        m.setObjective(quicksum(1 * wb_i_j[i][j] for i in range(n_c) for j in range(n_v)), GRB.MINIMIZE)
        m.optimize()

        if m.status == GRB.Status.OPTIMAL:

            inequalitys=[]
            for i in range(n_c):
                getvariables=[Symbol(domain.variables[j], REAL) for j in range(n_v)]
                rightside=Real(c_i[i].x)
                inequalitys.append(GE(rightside,Plus(Real(w_i_j[i][j].x)*getvariables[j] for j in range(n_v))))
            theory = And(t for t in inequalitys)

            return theory



        else:
            raise Z3Exception("L")

