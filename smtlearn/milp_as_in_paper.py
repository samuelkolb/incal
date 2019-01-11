

from gurobipy import *

import lplearing

from pysmt.shortcuts import Real, LE, Plus, Times, Symbol,And, GE
from pysmt.typing import REAL
from smt_print import pretty_print




def papermilp(domain,data,numberofcosntraints):
    m = Model("milppaper")
    #constants
    n_c=numberofcosntraints #number of constraint #i
    n_v=len(model.domain.real_vars) #number of varaibles #j
    n_e=len(data)#number of examples len(data) #k
    bigm=10000000
    ep=1
    wmax=1000
    cmax=1000
    c_0=1
    c_j=c_0+(1-c_0)#add compelxity here


    v_j_k = [[row[v] for v in model.domain.real_vars] for row, _  in data]
    #v_j_k = [[row[v] for v in var] for row, _  in data]
    labels = [row[1] for row in data]

    #variables


    c_i = [m.addVar(vtype=GRB.CONTINUOUS, lb=-cmax, ub=cmax,name="c_i({i})".format(i=i))for i in range(n_c)]

    cb_i = [m.addVar(vtype=GRB.BINARY,name="cb_i({i})".format(i=i))for i in range(n_c)]

    w_i_j=[[m.addVar(vtype=GRB.CONTINUOUS, lb=-wmax, ub=wmax,name="w_i_j({i},{j})".format(i=i,j=j))for j in range(n_v)] for i in range(n_c)]

    wb_i_j=[[m.addVar(vtype=GRB.BINARY,name="wb_i_j({i},{j})".format(i=i,j=j))for j in range(n_v)] for i in range(n_c)]

    s_k_i=[[m.addVar(vtype=GRB.BINARY,name="s_k_i({k},{i})".format(k=k,i=i))for i in range(n_c)] for k in range(n_e)]


    wf_i_j=[[m.addVar(vtype=GRB.BINARY,name="wf_i_j({i},{j})".format(i=i,j=j))for j in range(n_v)] for i in range(n_c)]
    wl_i_j=[[m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=wmax,name="wl_i_j({i},{j})".format(i=i,j=j))for j in range(n_v)] for i in range(n_c)]
    wu_i_j=[[m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=wmax,name="wu_i_j({i},{j})".format(i=i,j=j))for j in range(n_v)] for i in range(n_c)]

    cf_i = [m.addVar(vtype=GRB.BINARY,name="cf_i({i})".format(i=i))for i in range(n_c)]
    cl_i = [m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=cmax,name="cl_i({i})".format(i=i))for i in range(n_c)]
    cu_i = [m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=cmax,name="cu_i({i})".format(i=i))for i in range(n_c)]

    #constraints


    #5 sum(wij*xkj<=ci) A positives
    for k in range(n_e):
        if data[k][1]:
            for i in range(n_c): # on per postive exxample and number of constarints
                m.addConstr(sum(w_i_j[i][j]*v_j_k[k][j] for j in range(n_v)) <= c_i[i], name="c1({k},{i})".format(k=k,i=i))
    #6 sum(wij*sjk>m*ski-M+ci+e) A negatives
    for k in range(n_e):
        if not data[k][1]:
            for i in range(n_c): # on per postive exxample and number of constarints
                m.addConstr(sum(w_i_j[i][j]*v_j_k[k][j] for j in range(n_v)) >= bigm*s_k_i[k][i]- bigm+ c_i[i]+ep, name="c2({k},{i})".format(k=k,i=i))
    #7 sum(ski>=1) A negaives
    for k in range(n_e):# on per negative example
        if not data[k][1]:
            m.addConstr(sum(s_k_i[k][i]for i in range(n_c)) >= 1, name="c3({k})".format(k=k))
    #8 sum wbij>=cbi paper 18
    for i in range(n_c):
        m.addConstr(sum(wb_i_j[i][j] for j in range(n_v))>=cb_i[i],name="c4({i})".format(i=i))
    #5wij<=wmax*wbij
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


    #10 from paper
    for i in range(n_c):
        for j in range(n_v):
            m.addConstr(w_i_j[i][j]==wu_i_j[i][j]-wl_i_j[i][j]+1,name="cp10({i},{j})".format(i=i,j=j))
    #11 from paper
    for i in range(n_c):
        for j in range(n_v):
            m.addConstr(w_i_j[i][j]<=wmax-(wmax-1)*wf_i_j[i][j],name="cp11({i},{j})".format(i=i,j=j))

    #12 from paper
    for i in range(n_c):
        for j in range(n_v):
            m.addConstr(w_i_j[i][j]>=-wmax+(wmax+1)*wf_i_j[i][j],name="cp12({i},{j})".format(i=i,j=j))

    #15 from paper
    for i in range(n_c):
        m.addConstr(c_i[i]==cu_i[i]-cl_i[i]+1,name="cp15({i})".format(i=i))
    #16 from paper
    for i in range(n_c):
        m.addConstr(c_i[i]<=cmax-(cmax-1)*cf_i[i],name="cp16({i})".format(i=i))
    #17 from paper
    for i in range(n_c):
        m.addConstr(c_i[i]>=-cmax+(cmax+1)*cf_i[i],name="cp17({i})".format(i=i))

    #19 from paper
    m.addConstr(sum(wf_i_j[i][j] for i in range(n_c) for j in range(n_v))+sum(cf_i[i]for i in range(n_c))>=1,name="cp19({i})".format(i=i))

    #extra constraints for faster feasibility

    #paper 27
    for i in range(n_c):
        for j in range(n_v):
            m.addConstr(wb_i_j[i][j]>=wf_i_j[i][j],name="cp27({i},{j})".format(i=i,j=j))

    #paper 28
    for i in range(n_c):
        for j in range(n_v):
            m.addConstr(wl_i_j[i][j]+wu_i_j[i][j]<=bigm-bigm*wf_i_j[i][j],name="cp28({i},{j})".format(i=i,j=j))

    #paper 29
    for i in range(n_c):
        m.addConstr(cb_i[i]>=cf_i[i],name="cp29({i})".format(i=i))

    #paper 30
    for i in range(n_c):
        m.addConstr(cl_i[i]+cu_i[i]<=bigm-bigm*cf_i[i],name="cp30({i})".format(i=i))


    #+sum(c_0*cb_i[i] for i in range(n_c))

    m.setObjective(quicksum(1*wb_i_j[i][j] for i in range(n_c) for j in range(n_v))-1/1000*sum(sum(wf_i_j[i][j]for j in range(n_v))+cf_i[i] for i in range(n_c))+1/1000000*sum(sum(wl_i_j[i][j]+wu_i_j[i][j] for j in range(n_v))+cl_i[i]+cu_i[i]for i in range(n_c)), GRB.MINIMIZE)
    m.optimize()

    m.optimize()

    if m.status == GRB.Status.OPTIMAL:

        print("w_i_j[constrain][varaiable]")
        for i in range(n_c):
            #print('\t%s' %([w_i_j[i][j].varname for j in range(n_v) ]))
            print("{}<={}".format([[w_i_j[i][j].varname, w_i_j[i][j].x] for j in range(n_v)],c_i[i].x))

    inequalitys=[]
    for i in range(n_c):

        getvariables=[Symbol(domain.variables[j], REAL) for j in range(n_v)]
        rightside=Real(c_i[i].x)
        inequalitys.append(GE(rightside,Plus(Real(w_i_j[i][j].x)*getvariables[j] for j in range(n_v))))

    theory=And(t for t in inequalitys)


    m.write("/Users/Elias/Desktop/e.lp")

    return theory


model=lplearing.polutionreduction()
data=lplearing.sample_half_half(model,100)
x=papermilp(model.domain,data,3)