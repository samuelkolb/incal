
from pywmi.smt_math import LinearInequality
import csv
from parse import  nested_to_smt
from gurobipy import *

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

def importsmt(path):
    theories=[]
    with open(path) as csv_file:
         csv_reader = csv.reader(csv_file, delimiter=',')
         for row in csv_reader:
             theories.append(nested_to_smt(row[0]))
    return theories


def optimise_benchmark(smtformular, objectivevektor):

    m = Model("milp")
    m.setParam('OutputFlag', False)

    x=smtformular.get_atoms()
    inequalitylist=[]
    smtformular = smtformular.args()
    v_j_k = [[v for v in i.inequality_dict.values()] for i in inequalitylist]
    numbervaraiables = len(v_j_k[0]) - 1

    for f in range(len(smtformular)):
        inequalitylist.append(LinearInequality.from_smt(smtformular[f]))

    if len(x)==1:
        k=-v_j_k[1][0]
        v_j_k[0].append(k)
        v_j_k=[v_j_k[0]]

    x_i = [m.addVar(vtype=GRB.CONTINUOUS, name="x_i({i})".format(i=i)) for i in range(numbervaraiables)]

    m.setObjective(quicksum(x_i[i]*objectivevektor[i] for i in range(numbervaraiables)), GRB.MINIMIZE)

    for k in range(len(v_j_k)):
        m.addConstr(sum(x_i[i] * v_j_k[k][i] for i in range(numbervaraiables)) <= - v_j_k[k][-1],
                        name="c1({k})".format(k=k))


    if len(x) == 1:
        m.write("/Users/Elias/Desktop/NEW.lp")
    m.optimize()

    if m.status == GRB.Status.OPTIMAL:
        return m.objVal


polution=[8,10,7,6,11,9]
policeo=[170,160,175,180,195]



def scale_police(df):
    df["D"]=1+df["D"]/30610
    return df

def scale_pollution(df):
    df["D"] = 1+df["D"]/32
    return df


incalmilpresults=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/incal milp/polution/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, polution)
        incalmilpresults.append([j, r])

incalpsmtresults=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/incal smt/polution/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, polution)
        incalpsmtresults.append([j, r])

milpcsresults=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/results big milp/polution/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, polution)
        milpcsresults.append([j, r])

incalpsmtnorm=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/smt with extra constarin/polution/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, polution)
        incalpsmtnorm.append([j, r])

incalpconstantobjective=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/incal milp constant/polution/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, polution)
        incalpconstantobjective.append([j, r])

reducedmilpcs=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/results small milp/polution/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, polution)
        reducedmilpcs.append([j, r])

incalpmilp_pollution = pd.DataFrame(incalmilpresults, columns=['samplesize', 'objetivefunctionvalue'])
incalpsmt_pollution = pd.DataFrame(incalpsmtresults, columns=['samplesize', 'objetivefunctionvalue'])
milpcs_pollution = pd.DataFrame(milpcsresults, columns=['samplesize', 'objetivefunctionvalue'])

incalpsmtex_pollution = pd.DataFrame(incalpsmtnorm, columns=['samplesize', 'objetivefunctionvalue'])
incalpmilpconstant_pollution = pd.DataFrame(incalpconstantobjective, columns=['samplesize', 'objetivefunctionvalue'])
milpcsreduced_pollution = pd.DataFrame(reducedmilpcs, columns=['samplesize', 'objetivefunctionvalue'])


incalpmilp_pollution["D"]= incalpmilp_pollution["objetivefunctionvalue"] - 32
incalpsmt_pollution["D"]= incalpsmt_pollution["objetivefunctionvalue"] - 32
milpcs_pollution["D"]= milpcs_pollution["objetivefunctionvalue"] - 32

incalpsmtex_pollution["D"]= incalpsmtex_pollution["objetivefunctionvalue"] - 32
incalpmilpconstant_pollution["D"]= incalpmilpconstant_pollution["objetivefunctionvalue"] - 32
milpcsreduced_pollution["D"]= milpcsreduced_pollution["objetivefunctionvalue"] - 32

incalpmilp_pollution=scale_pollution(incalpmilp_pollution)
incalpsmt_pollution=scale_pollution(incalpsmt_pollution)
milpcs_pollution=scale_pollution(milpcs_pollution)

incalpsmtex_pollution=scale_pollution(incalpsmtex_pollution)
incalpmilpconstant_pollution=scale_pollution(incalpmilpconstant_pollution)
milpcsreduced_pollution=scale_pollution(milpcsreduced_pollution)

incalmilpresults=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/incal milp/police/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, policeo)
        incalmilpresults.append([j, r])

incalpsmtresults=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/incal smt/police/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, policeo)
        incalpsmtresults.append([j, r])

milpcsresults=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/results big milp/police/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, policeo)
        milpcsresults.append([j, r])

incalpsmtnorm=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/smt with extra constarin/police/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, policeo)
        incalpsmtnorm.append([j, r])

incalpconstantobjective=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/incal milp constant/police/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, policeo)
        incalpconstantobjective.append([j, r])

reducedmilpcs=[]
for j in [20,30,40,50,100,200,300,500,1000,2000]:
    path="/Users/Elias/Documents/GitHub/smtlearn/output/results small milp/police/theories_learned/S:{}.csv".format(j)
    smtformulars=importsmt(path)
    for i in smtformulars:
        r=optimise_benchmark(i, policeo)
        reducedmilpcs.append([j, r])

incalpmilp_police = pd.DataFrame(incalmilpresults, columns=['samplesize', 'objetivefunctionvalue'])
incalpsmt_police = pd.DataFrame(incalpsmtresults, columns=['samplesize', 'objetivefunctionvalue'])
milpcs_police = pd.DataFrame(milpcsresults, columns=['samplesize', 'objetivefunctionvalue'])

incalpsmtnorm_police = pd.DataFrame(incalpsmtnorm, columns=['samplesize', 'objetivefunctionvalue'])
incalpmilpconstant_police = pd.DataFrame(incalpconstantobjective, columns=['samplesize', 'objetivefunctionvalue'])
milpcsreduced_police = pd.DataFrame(reducedmilpcs, columns=['samplesize', 'objetivefunctionvalue'])

incalpmilp_police["D"]= incalpmilp_police["objetivefunctionvalue"] * 200 - 30610
incalpsmt_police["D"]= incalpsmt_police["objetivefunctionvalue"] * 200 - 30610
milpcs_police["D"]= milpcs_police["objetivefunctionvalue"] * 200 - 30610

incalpmilp_police=scale_police(incalpmilp_police)
incalpsmt_police=scale_police(incalpsmt_police)
milpcs_police=scale_police(milpcs_police)

incalpsmtnorm_police["D"]= incalpsmtnorm_police["objetivefunctionvalue"] * 200 - 30610
incalpmilpconstant_police["D"]= incalpmilpconstant_police["objetivefunctionvalue"] * 200 - 30610
milpcsreduced_police["D"]= milpcsreduced_police["objetivefunctionvalue"] * 200 - 30610


incalpsmtnorm_police=scale_police(incalpsmtnorm_police)
incalpmilpconstant_police=scale_police(incalpmilpconstant_police)
milpcsreduced_police=scale_police(milpcsreduced_police)





def std(data):
    datagroup = data.groupby(["samplesize"])
    temp = []
    for [label, df] in datagroup:
        timestd = np.std(df["D"], axis=0) / np.sqrt(df["D"].shape[0])
        mean=np.mean(df["D"],axis=0)
        temp.append([label, timestd,mean])
    standartdeviation=pd.DataFrame(temp,columns=["samplesize","std",'mean'])

    return standartdeviation


incalpmilp_pollution_std=std(incalpmilp_pollution)
incalpsmt_pollution_std=std(incalpsmt_pollution)
milpcs_pollution_std=std(milpcs_pollution)
incalpmilp_police_std=std(incalpmilp_police)
incalpsmt_police_std=std(incalpsmt_police)
milpcs_police_std=std(milpcs_police)

incalpsmtex_pollution_std=std(incalpsmtex_pollution)
incalpmilpconstant_pollution_std=std(incalpmilpconstant_pollution)
milpcsreduced_pollution_std=std(milpcsreduced_pollution)
incalpsmtnorm_police_std=std(incalpsmtnorm_police)
incalpmilpconstant_police_std=std(incalpmilpconstant_police)
milpcsreduced_police_std=std(milpcsreduced_police)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8),sharey='row',constrained_layout=True)
fig.suptitle("Difference between true and found objective value")

incalpmilp_pollution.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[0], label="IncaLP(MILP)", color="#1f77b4")
ax[0].fill_between(incalpmilp_pollution_std["samplesize"], incalpmilp_pollution_std["mean"] - incalpmilp_pollution_std["std"], incalpmilp_pollution_std["mean"] + incalpmilp_pollution_std["std"], color="#1f77b4", alpha=0.35, linewidth=0)

incalpsmt_pollution.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[0], label="IncaLP(SMT)", color="#ff7f0e")
ax[0].fill_between(incalpsmt_pollution_std["samplesize"], incalpsmt_pollution_std["mean"] - incalpsmt_pollution_std["std"], incalpsmt_pollution_std["mean"] + incalpsmt_pollution_std["std"], color="#ff7f0e", alpha=0.35, linewidth=0)

milpcs_pollution.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[0], label="MILPCS", color="#2ca02c")
ax[0].fill_between(milpcs_pollution_std["samplesize"], milpcs_pollution_std["mean"] - milpcs_pollution_std["std"], milpcs_pollution_std["mean"] + milpcs_pollution_std["std"], color="#2ca02c", alpha=0.35, linewidth=0)

incalpsmtex_pollution.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[0], label="IncaLP(SMT) \n Extended", color="#8B008B")
ax[0].fill_between(incalpsmtex_pollution_std["samplesize"], incalpsmtex_pollution_std["mean"] - incalpsmtex_pollution_std["std"], incalpsmtex_pollution_std["mean"] + incalpsmtex_pollution_std["std"], color="#8B008B", alpha=0.35, linewidth=0)

incalpmilpconstant_pollution.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[0], label="IncaLP(MILP) \n constant objective", color="#FF0000")
ax[0].fill_between(incalpsmtex_pollution_std["samplesize"], incalpmilpconstant_pollution_std["mean"] - incalpmilpconstant_pollution_std["std"], incalpmilpconstant_pollution_std["mean"] + incalpmilpconstant_pollution_std["std"], color="#FF0000", alpha=0.35, linewidth=0)

milpcsreduced_pollution.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[0], label="MILPCS reduced objective", color="#FF00FF")
ax[0].fill_between(incalpsmtex_pollution_std["samplesize"], milpcsreduced_pollution_std["mean"] - milpcsreduced_pollution_std["std"], milpcsreduced_pollution_std["mean"] + milpcsreduced_pollution_std["std"], color="#FF00FF", alpha=0.35, linewidth=0)



incalpmilp_police.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[1], label="IncaLP(MILP)")
ax[1].fill_between(incalpmilp_police_std["samplesize"], incalpmilp_police_std["mean"] - incalpmilp_police_std["std"], incalpmilp_police_std["mean"] + incalpmilp_police_std["std"], color="gray", alpha=0.35, linewidth=0)

incalpsmt_police.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[1], label="IncaLP(SMT)", color="#ff7f0e")
ax[1].fill_between(incalpsmt_police_std["samplesize"], incalpsmt_police_std["mean"] - incalpsmt_police_std["std"], incalpsmt_police_std["mean"] + incalpsmt_police_std["std"], color="#ff7f0e", alpha=0.35, linewidth=0)

milpcs_police.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[1], label="MILPCS", color="#2ca02c")
ax[1].fill_between(milpcs_police_std["samplesize"], milpcs_police_std["mean"] - milpcs_police_std["std"], milpcs_police_std["mean"] + milpcs_police_std["std"], color="#2ca02c", alpha=0.35, linewidth=0)

incalpsmtnorm_police.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[1], label="IncaLP(SMT) \n Extended", color="#8B008B")
ax[1].fill_between(incalpsmtnorm_police_std["samplesize"], incalpsmtnorm_police_std["mean"] - incalpsmtnorm_police_std["std"], incalpsmtnorm_police_std["mean"] + incalpsmtnorm_police_std["std"], color="#8B008B", alpha=0.35, linewidth=0)

incalpmilpconstant_police.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[1], label="IncaLP(MILP) \n constant objective", color="#FF0000")
ax[1].fill_between(incalpmilpconstant_police_std["samplesize"], incalpmilpconstant_police_std["mean"] - incalpmilpconstant_police_std["std"], incalpmilpconstant_police_std["mean"] + incalpmilpconstant_police_std["std"], color="#FF0000", alpha=0.35, linewidth=0)

milpcsreduced_police.groupby(['samplesize']).mean()["D"].plot(x="samplesize", y="D", ax=ax[1], label="MILPCS reduced objective", color="#FF00FF")
ax[1].fill_between(milpcsreduced_police_std["samplesize"], milpcsreduced_police_std["mean"] - milpcsreduced_police_std["std"], milpcsreduced_police_std["mean"] + milpcsreduced_police_std["std"], color="#FF00FF", alpha=0.35, linewidth=0)


ax[0].set(ylabel="Mean ratio")
ax[1].set(ylabel="Mean ratio")
ax[0].set(xlabel="Sample size")
ax[1].set(xlabel="Sample size")

ax[0].set_title("Pollution")
ax[1].set_title("Police")


ax[0].axhline(y=1, color='#000000', linestyle='-')
ax[1].axhline(y=1, color='#000000', linestyle='-')
ax[1].legend(ncol=2)

dir_name = "/Users/Elias/Documents/GitHub/smtlearn/Plotts/"
plotname="regret"

fig.savefig(dir_name+plotname+".png",bbox_inches='tight', pad_inches=0)