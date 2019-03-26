from pywmi.engines.latte_backend import LatteIntegrator

#from milp import  rockingthemilp
#from milp_as_in_paper import papermilp
import pywmi
#from smt_print import pretty_print

#from lplearing import polutionreduction, cuben,sample_half_half,learn_parameter_free
#from pysmt.shortcuts import Real,Ite
import pysmt.shortcuts as smt
#from problem import Domain, Problem
import csv
from parse import smt_to_nested, nested_to_smt
from smt_print import pretty_print
import pandas as pd
from pywmi import XaddEngine, RejectionEngine, Domain
import pysmt.shortcuts as smt

import logging

#logging.basicConfig(level=logging.DEBUG)
#model=polutionreduction()
#sample=sample_half_half(model,20)

#theory_one, x, y=learn_parameter_free(model, sample, 10)
#theory_two=rockingthemilp(model.domain, sample, 3)
#theory_three,n=papermilp(model.domain, sample, 13)

#print("Learned theory:\n{}".format(pretty_print(learned_theory)))
#sample_count=1000000


#tpr1 = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
 #                                     sample_count).compute_probability(theory_one)
#tnr1 = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
 #                                     sample_count).compute_probability(~theory_one)


#tpr2 = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
 #                                     sample_count).compute_probability(theory_two)
#tnr2 = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
 #                                     sample_count).compute_probability(~theory_two)

#tpr3 = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
 #                                     sample_count).compute_probability(theory_three)
#tnr3 = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
 #                                   sample_count).compute_probability(~theory_three)


#tp = pywmi.XaddEngine(model.domain, model.theory, smt.Real(1.0)).compute_probability(theory_one)
#print(model.domain)
#print(pretty_print(model.theory))
#support = model.theory & model.domain.get_bounds()
# x=XaddEngine(model.domain, support, smt.Real(1.0), mode="resolve").compute_volume()
#print(x)
#exit()
#domain = Domain.make(["a", "b"], ["x", "y"], [(0, 1), (0, 1)])
#a=domain.get_symbol("a")
#b=domain.get_symbol("b")
#x=domain.get_symbol("x")
#y=domain.get_symbol("y")

#support = (a | b) & (~a | ~b) & (x <= y) & domain.get_bounds()
#query = x <= y / 2
#weight_function = smt.Ite(a, smt.Real(0.2), smt.Real(0.8)) * (smt.Ite(x <= 0.5, smt.Real(0.2), 0.2 + y) + smt.Real(0.1))
#xadd_engine = XaddEngine(domain, support, smt.Real(1.0), mode="resolve")
from pysmt.shortcuts import *

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

def normilisation_max(theory):
    inequalitys = []

    if len(theory.get_atoms()) == 1:
        constant = theory.arg(1)
        leftside = theory.arg(0)
        m = [leftside.arg(j).arg(0).constant_value() for j in range(len(leftside.args()))]
        m.append(constant.constant_value())
        maxallc = max(m)

        if maxallc > 0:

            le = Real(constant.constant_value() / maxallc)

            inequalitys.append(GE(le, Plus(
                [leftside.arg(j).arg(0).constant_value() / maxallc * leftside.arg(j).arg(1) for j in
                 range(len(leftside.args()))])))
        elif maxallc < 0:

            le = Real(constant.constant_value() / maxallc)
            inequalitys.append(LE(le, Plus(
                [leftside.arg(j).arg(0).constant_value() / maxallc * leftside.arg(j).arg(1) for j in
                 range(len(leftside.args()))])))
        else:
            return theory

    else:
        for i in range(len(theory.get_atoms())):
            constant = theory.arg(i).arg(1)
            leftside = theory.arg(i).arg(0)
            m = [leftside.arg(j).arg(0).constant_value() for j in range(len(leftside.args()))]
            m.append(constant.constant_value())
            maxallc = max(m)
            if maxallc ==0:
                m=list(filter(lambda a: a != 0, m))
                maxallc=max(m)


            if maxallc > 0:


                le = Real(constant.constant_value() / maxallc)

                inequalitys.append(GE(le, Plus(
                    [leftside.arg(j).arg(0).constant_value() / maxallc * leftside.arg(j).arg(1) for j
                     in range(len(leftside.args()))])))
            elif maxallc < 0:

                le = Real(constant.constant_value() / maxallc)

                inequalitys.append(LE(le, Plus(
                    [leftside.arg(j).arg(0).constant_value() / maxallc * leftside.arg(j).arg(1) for j
                     in range(len(leftside.args()))])))
            else:

                le = Real(constant.constant_value())

                inequalitys.append(GE(le, Plus([leftside.arg(j).arg(0).constant_value() * leftside.arg(j).arg(1) for j in
                                                range(len(leftside.args()))])))

    normalisedtheory = And(inequalitys)
    return normalisedtheory

# with open("/Users/Elias/Documents/GitHub/smtlearn/output/polution/theories_learned/S:500.csv") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         y=nested_to_smt(row[0])
#         x=normilisation_rightsideto1(y)
#         z=normilisation_max(y)
#         print(len(y.get_atoms()))
#         print(pretty_print(y))
#         print(pretty_print(x))
#         print(pretty_print(z))


from pysmt.oracles import AtomsOracle #get_atoms
from pysmt.oracles import FreeVarsOracle

#imd="/Users/Elias/Documents/GitHub/smtlearn/output/small milp/cuben/theories_learned/D:"
#imd2="S:"
#imd3=".csv"
#for j in [2,3]:
 #   for i in [20,30,40,50,100,200,300,400,500]:
  #      with open(imd+str(j)+imd2+str(i)+imd3) as csv_file:
   #         csv_reader = csv.reader(csv_file, delimiter=',')
    #        line_count = 0
     #       for row in csv_reader:
       #         y=nested_to_smt(row[0])
       #         l.append(len(y.get_atoms()))
        #        l2.append(2*j)
                #l2.append(j*(j-1)+1 )

#for j in [4]:
 #   for i in [20,30,40,50,100,200,300]:
  #      with open(imd+str(j)+imd2+str(i)+imd3) as csv_file:
   #         csv_reader = csv.reader(csv_file, delimiter=',')
    #        line_count = 0
     #       for row in csv_reader:
      #          y=nested_to_smt(row[0])
       #         l.append(len(y.get_atoms()))
                #l2.append(j * (j - 1) + 1)
        #        l2.append(2 * j)

#solver="small milp"
#instance="cuben"
#importfile="Cuben overall small MILP.csv"
#dir_name = "../output/{}/{}/".format(solver,instance)
#data = pd.read_csv(dir_name + importfile,header=0,index_col=0)#, names=["run","dimension","samplesize","time","actual","synthesized","tpr","tnr"],index_col=0 )

#data["synthesized"]=l
#data["actual"]=l2

#data.to_csv(dir_name + importfile)


     #   tpr1 = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
        #                                     sample_count).compute_probability(y)
      #  tnr1 = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
         #                                    sample_count).compute_probability(~y)
       # print(tpr1,tnr1)

import json
import os


def load(filename):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump({"files": {}}, f)

    with open(filename, "r") as f:
        flat = json.load(f)
    return flat


# len(os.listdir("/Users/Elias/Documents/GitHub/smtlearn/output/polution/50/6688621"))
# with open("/Users/Elias/Documents/GitHub/smtlearn/output/results high H Milp/police/2/358588/observer200:5:2.csv") as f:
#     data = [json.loads(line) for line in f]
# last=data[:-1]
# t=nested_to_smt(last["theory"])

#l=[]
#for i in data:
 #   if "solving_time" in i.keys():
 #       l.append(i["solving_time"])

def importlogforonessed(generall):
    a=[]
    b=[]
    c=[]
    for filename in os.listdir(generall):
        with open(generall+"/"+filename) as f:
            data = [json.loads(line) for line in f]
            b.append(len(data))
            l=[]
            ll=[]
            for i in data:
                if "selection_time" in i.keys():
                   l.append(i["selection_time"])
                if "solving_time" in i.keys():
                    ll.append(i["solving_time"])
            a.append(sum(l))
            c.append(sum(ll))
    return sum(a),sum(b),sum(c)



def loadperh(path):
    d=[]
    for filename in os.listdir(path):
        with open(path + "/" + filename) as f:
            data = [json.loads(line) for line in f]
            selection_time=[]
            solving_time=[]
            for i in data:
                if "selection_time" in i.keys():
                    selection_time.append(i["selection_time"])
                if "solving_time" in i.keys():
                    solving_time.append(i["solving_time"])

            i=int(filename[-5])
            d.append([i,sum(solving_time)])
    return d

def whole(path):
    l=[]
    for filename in os.listdir(path):
        l=l+loadperh(path+filename)

    df = pd.DataFrame(l, columns=["H","solving_time"])
    return df

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
xx=whole("/Users/Elias/Documents/GitHub/smtlearn/output/smt/polution/1000/")
xx['ID']=1
yy=whole("/Users/Elias/Documents/GitHub/smtlearn/output/incal milp/polution/1000/")
yy['ID']=0
frames = [xx,yy]

result = pd.concat(frames)
plt.subplots(figsize=(4,3))
sns.boxplot(x="H", y="solving_time",hue="ID" ,data=result, palette="PRGn")
plt.tight_layout()

plt.subplots(figsize=(4,3))
sns.boxplot(x="H", y="solving_time" ,data=xx, palette="PRGn")
plt.tight_layout()



#selection_time
#solving_time

#list of all seeds
#os.listdir("/Users/Elias/Documents/GitHub/smtlearn/output/polution/300")


def metrix(p):
    l=[]
    l2=[]
    l3=[]
    labels=[]
    for j in [2, 4, 8, 16, 32, 64]:
        d=p+str(j)
        t=[]
        t2=[]
        t3=[]
        labels.append(j)
        for i in os.listdir(d):
            g=d+"/"+str(i)
            a,b,c=importlogforonessed(g)
            t.append(a)
            t2.append(b)
            t3.append(c)
        l.append(sum(t)/len(t))
        l2.append((sum(t2)/len(t2)))
        l3.append(sum(t3)/len(t3))

    return l,l2,l3

def metrixx(p):
    l=[]
    l2=[]
    l3=[]
    labels=[]
    for j in [20,30,40,50,100,200,300,500]:
        d=p+str(j)
        t=[]
        t2=[]
        t3=[]
        labels.append(j)
        for i in os.listdir(d):
            g=d+"/"+str(i)
            a,b,c=importlogforonessed(g)
            t.append(a)
            t2.append(b)
            t3.append(c)
        l.append(sum(t)/len(t))
        l2.append((sum(t2)/len(t2))+20)
        l3.append(sum(t3)/len(t3))

    return l,l2,l3

# p="/Users/Elias/Documents/GitHub/smtlearn/output/results high H smt/polution/"
# l,l2,l3=metrix(p)
#
# p="/Users/Elias/Documents/GitHub/smtlearn/output/results high H Milp N/polution/"
# l1,l21,l31=metrix(p)
#
p="/Users/Elias/Documents/GitHub/smtlearn/output/incal milp/simplexn/"
a,b,c=metrixx(p)

p="/Users/Elias/Documents/GitHub/smtlearn/output/smt/simplexn/"
a1,b1,c1=metrixx(p)

x=[20,30,40,50,100,200,300,500]
plt.subplots(figsize=(4,3))
plt.plot(x, b, 'ro',label="Incremental:Milp")
plt.plot(x, b1,  'g^',label="Incremental:Smt")

plt.plot([0, 500], [0, 500], ls="--")
plt.legend()
plt.ylabel('Number of selected examples')
plt.xlabel('Sample size')
plt.tight_layout()


# import numpy as np
# labels=[2, 4, 8, 16, 32, 64]
# plt.figure(1)
# o=["2", "4", "8", "16", "32", "64"]
# y=np.arange(len(labels))
# plt.bar(y, l, align='center', alpha=0.5,label="smt")
# plt.bar(y, l1, align='center', alpha=0.5,label="milp")
# plt.legend()
# plt.xticks(y, o)
# plt.ylabel('selection_time')
# plt.title('police problem with diffrent H')
#
# plt.figure(2)
# o=["2", "4", "8", "16", "32", "64"]
# y=np.arange(len(labels))
# plt.bar(y, l2, align='center', alpha=0.5,label="smt")
# plt.bar(y, l21, align='center', alpha=0.5,label="milp")
# plt.legend()
# plt.xticks(y, o)
# plt.ylabel('active examples')
# plt.title('police problem with diffrent H')
#
# plt.figure(3)
# o=["2", "4", "8", "16", "32", "64"]
# y=np.arange(len(labels))
# plt.bar(y, l3, align='center', alpha=0.5,label="smt")
# plt.bar(y, l31, align='center', alpha=0.5,label="milp")
# plt.legend()
# plt.xticks(y, o)
# plt.ylabel('solving_time')
# plt.title('police problem with diffrent H')