
from pywmi.smt_math import LinearInequality
import csv
import pysmt.shortcuts as smt
from parse import smt_to_nested, nested_to_smt
from gurobipy import *
from smt_print import pretty_print
from lp_problems import cuben, simplexn, polutionreduction, police2

#import all found functions for a sample size
def importsmt(path):
    y=[]
    with open(path) as csv_file:
         csv_reader = csv.reader(csv_file, delimiter=',')
         line_count = 0
         for row in csv_reader:
             y.append(nested_to_smt(row[0]))
    return y


def optimise(smtformular,objectivevektor):

    inequalitylist=[]
    smtformular=smtformular.args()
    print(len(smtformular))


    for f in range(len(smtformular)):
        inequalitylist.append(LinearInequality.from_smt(smtformular[f]))


    v_j_k = [[v for v in i.inequality_dict.values()] for i in inequalitylist]

    #for smt_poilce
    #print(v_j_k)
    #for i in range(len(v_j_k)):
     #   v_j_k[i]=v_j_k[i][1:] + [v_j_k[i][0]]
    #print(v_j_k)

    numbervaraiables=len(v_j_k[1])-1

    m = Model("milp")
    #m.setParam('TimeLimit', 60 * 90)
    m.setParam('OutputFlag', False)

    x_i = [m.addVar(vtype=GRB.CONTINUOUS, name="x_i({i})".format(i=i)) for i in range(numbervaraiables)]

    m.setObjective(quicksum(x_i[i]*objectivevektor[i] for i in range(numbervaraiables)), GRB.MINIMIZE)
    #m.setObjective(2,GRB.MINIMIZE)

    for k in range(len(v_j_k)):
        m.addConstr(sum(x_i[i] * v_j_k[k][i] for i in range(numbervaraiables)) <= - v_j_k[k][-1],
                        name="c1({k})".format(k=k))

        #name = "c4({i})".format(i=i))

    m.write("/Users/Elias/Desktop/NEW.lp")
    m.optimize()

    if m.status == GRB.Status.OPTIMAL:
        print("{}".format([[x_i[i].varname, x_i[i].x] for i in range(numbervaraiables)]))
        print('\nCost: %g' % m.objVal)
        return m.objVal#200 for police

l=[]
polution=[8,10,7,6,11,9]
policeo=[170,160,175,180,195]

#optimise(smtformulars[0],polution)
#x=police2()
#l=optimise(smtformulars[0],policeo)
#l=optimise(x.theory,policeo)
import pandas as pd
import matplotlib.pyplot as plt


milpobjectiveresults=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/incal milp/polution/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise(i,polution)
        milpobjectiveresults.append([j,r])

smtobjectiveresults=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/smt/polution/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise(i,polution)
        smtobjectiveresults.append([j,r])

df1 = pd.DataFrame(milpobjectiveresults,columns=['samplesize', 'objetivefunctionvalue'])
df2 = pd.DataFrame(smtobjectiveresults,columns=['samplesize', 'objetivefunctionvalue'])

df1["D"]=df1["objetivefunctionvalue"]-32.16
df2["D"]=df2["objetivefunctionvalue"]-32.16

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
fig.suptitle("Polution")

df1.groupby(['samplesize']).mean()["objetivefunctionvalue"].plot(x="samplesize", y="objetivefunctionvalue", ax=ax[0], label="milp")
df2.groupby(['samplesize']).mean()["objetivefunctionvalue"].plot(x="samplesize", y="objetivefunctionvalue", ax=ax[0], label="smt")

df1.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="objetivefunctionvalue", ax=ax[1], label="milp")
df2.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="objetivefunctionvalue", ax=ax[1], label="smt")

ax[0].set(ylabel="Value")
ax[1].set(ylabel="Difference")
ax[0].axhline(y=32, color='r', linestyle='-')
ax[1].axhline(y=0, color='r', linestyle='-')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")