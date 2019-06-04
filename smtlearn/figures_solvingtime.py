
import pandas as pd
from pysmt.shortcuts import *
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

def normalisation_by_max(theory):
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




def load(filename):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump({"files": {}}, f)

    with open(filename, "r") as f:
        flat = json.load(f)
    return flat


def importlogforonessed(generall):
    overallselctiontime=[]
    encoded=[]
    solvingtime=[]
    if generall == "/Users/Elias/Documents/GitHub/smtlearn/output/incal smt/polution/2000/.DS_Store":
        generall="/Users/Elias/Documents/GitHub/smtlearn/output/incal smt/polution/2000/"
    fl=os.listdir(generall)
    for filename in fl:
        if not filename.startswith('.')and os.path.isfile(os.path.join(generall, filename)):
            with open(generall+"/"+filename) as f:
                data = [json.loads(line) for line in f]
                encoded.append(len(data))
                tempselctiontime=[]
                tempsolvingtime=[]
                for i in data:
                    if "selection_time" in i.keys():
                       tempselctiontime.append(i["selection_time"])
                    if "solving_time" in i.keys():
                        tempsolvingtime.append(i["solving_time"])
                overallselctiontime.append(sum(tempselctiontime))
                solvingtime.append(sum(tempsolvingtime))
    return sum(overallselctiontime),sum(encoded),sum(solvingtime)


def loadperh(path):
    solvingtime=[]
    for filename in os.listdir(path):
        with open(path + "/" + filename) as f:
            data = [json.loads(line) for line in f]
            selection_time=[]
            solving_time=[]
            for m in data:
                if "selection_time" in m.keys():
                    selection_time.append(m["selection_time"])
                if "solving_time" in m.keys():
                    solving_time.append(m["solving_time"])

            m=int(filename[-5])
            solvingtime.append([m,sum(solving_time)])
    return solvingtime


def whole(path):
    temp=[]
    for filename in os.listdir(path):
        temp=temp+loadperh(path+filename)
    df = pd.DataFrame(temp, columns=["H","solving_time"])
    return df


smtsolvingtimepollution=whole("/Users/Elias/Documents/GitHub/smtlearn/output/incal smt/polution/500/")
smtsolvingtimepollution['ID']=1
milpsolvingtimepollution=whole("/Users/Elias/Documents/GitHub/smtlearn/output/incal milp/polution/500/")
milpsolvingtimepollution['ID']=0

smtsolvingtimepolice=whole("/Users/Elias/Documents/GitHub/smtlearn/output/incal smt/polution/500/")
smtsolvingtimepolice['ID']=1
milpsolvingtimepolice=whole("/Users/Elias/Documents/GitHub/smtlearn/output/incal milp/polution/500/")
milpsolvingtimepolice['ID']=0


pollutionsolvingtime = pd.concat([smtsolvingtimepollution, milpsolvingtimepollution])
policesolvingtime = pd.concat([smtsolvingtimepolice, milpsolvingtimepolice])



fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(12,3))

fig.suptitle("Solving time spend")

sns.boxplot(x="H", y="solving_time", hue="ID", data=pollutionsolvingtime, palette="PRGn", ax=ax[0])
sns.boxplot(x="H", y="solving_time", hue="ID", data=policesolvingtime, palette="PRGn", ax=ax[1])

ax[0].set(ylabel="Solving time (s)")
ax[1].set(ylabel="Solving time (s)")

ax[0].set(xlabel="H")
ax[1].set(xlabel="H")

ax[0].set_title("Pollution")
ax[1].set_title("Police")
#labels = ["Incal(SMT)","IncalP(MILP)"]

ax[0].get_legend().remove()
ax[1].get_legend().remove()

handles, _ = ax[0].get_legend_handles_labels()
plt.legend(handles, ["Incal(SMT)","IncalP(MILP)"],bbox_to_anchor=(1.04, 1), loc="upper left")

plt.tight_layout()
dir_name = "/Users/Elias/Documents/GitHub/smtlearn/Plotts/"
plotname="timespend"
fig.savefig(dir_name+plotname+".eps",format='eps')






def collectruninfo(p):
    selectiontime=[]
    encodedexamples=[]
    solvingtime=[]
    labels=[]
    for j in [20,30,40,50,100,200,300,500,1000,2000]:
        d=p+str(j)
        currentselectiontime=[]
        currentencoded=[]
        currentsolvingtime=[]
        labels.append(j)
        for i in os.listdir(d):
            g=d+"/"+str(i)
            a,b,c=importlogforonessed(g)
            currentselectiontime.append(a)
            currentencoded.append(b)
            currentsolvingtime.append(c)
        selectiontime.append(sum(currentselectiontime)/len(currentselectiontime))
        encodedexamples.append((sum(currentencoded)/len(currentencoded))+20)
        solvingtime.append(sum(currentsolvingtime)/len(currentsolvingtime))

    return selectiontime,encodedexamples,solvingtime



p="/Users/Elias/Documents/GitHub/smtlearn/output/incal milp/polution/"
a,b,c=collectruninfo(p)

p="/Users/Elias/Documents/GitHub/smtlearn/output/incal smt/polution/"
a1,b1,c1=collectruninfo(p)



fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(12,3))
fig.suptitle("Pollution")
x=[20,30,40,50,100,200,300,500,1000,2000]

ax[0].plot(x, b, 'ro',label="IncalP(MILP)",color="#1f77b4")
ax[0].plot(x, b1,  'g^',label="IncalP(SMT)",color="#ff7f0e")

col=["#1f77b4","#ff7f0e","#1f77b4","#ff7f0e","#1f77b4","#ff7f0e","#1f77b4","#ff7f0e","#1f77b4","#ff7f0e","#1f77b4","#ff7f0e","#1f77b4","#ff7f0e"]
box=sns.boxplot(x="H", y="solving_time", hue="ID", data=policesolvingtime, palette="PRGn")#,ax=ax[1])
for i in range(len(box.artists)):
    m=box.artists[i]
    m.set_facecolor(col[i])

ax[0].plot([0, 2000], [0, 2000], ls="--")


ax[0].set(ylabel="Average number of examples encoded")
ax[1].set(ylabel="Solving time (s)")



ax[0].set(xlabel="Sample size")
ax[1].set(xlabel="m")

ax[0].set_title("Examples encoded during IncaLP")
ax[1].set_title("Spread of solving time")
ax[1].get_legend().remove()

labels = ["IncaLP(MILP)","IncaLP(SMT)"]
ax[0].legend(labels)

plt.tight_layout()
dir_name = "/Users/Elias/Documents/GitHub/smtlearn/Plotts/"
plotname="example use"
fig.savefig(dir_name+plotname+".png",bbox_inches='tight', pad_inches=0)


