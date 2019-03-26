import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#solver="incal milp"
solver="syn smt"
instance="2"
importfile="Results[2, 3, 4, 5, 6, 7, 8, 9, 10][20, 30, 40, 50, 100, 200, 300, 500].csv"
#importfile="Resultspolution[2, 4, 8, 16, 32, 64].csv"
dir_name = "../output/{}/{}/".format(solver,instance)

data=pd.read_csv(dir_name+importfile,index_col=0,names=["dimension","samplesize","time","actual","synthesized","tpr","tnr"])


averageoverseed=data.groupby(["dimension","samplesize"]).mean()
std=data.groupby(["dimension","samplesize"]).std()

#del averageoverseed["run"]
#del std["run"]
averageoverseed["Constrain_difference:Mean"]=averageoverseed["synthesized"]-averageoverseed["actual"]
std["Constrain_difference:Std"]=std["synthesized"]-std["actual"]

averageoverseed.columns=["Time:Mean","ActualD","Found:mean","TPR:Mean","TNR:Mean","Constrain_difference:Mean"]
std.columns=["Time:STD","Actual","Found:STD","TPR:STD","TNR:STD","Constrain_difference:Std"]

dfy=pd.concat([averageoverseed, std],axis=1)

del dfy["ActualD"]
del dfy["Found:mean"]
del dfy["Found:STD"]


dfy=dfy[['Time:Mean','Time:STD',"Constrain_difference:Mean" ,"Constrain_difference:Std", 'TPR:Mean', 'TPR:STD', 'TNR:Mean', 'TNR:STD']]

colsto1 = ['Time:Mean','Time:STD']
dfy[colsto1] = dfy[colsto1].round(1)
colsto2 = ['TPR:Mean', 'TPR:STD', 'TNR:Mean', 'TNR:STD']
dfy[colsto2] = dfy[colsto2].round(2)


dfy.to_csv(dir_name+instance+"Mean results.csv")



fig, ax = plt.subplots(figsize=(4,3))
data.groupby(['dimension','samplesize']).mean()["time"].unstack(level=0).plot(ax=ax,title=solver+instance+"Halfspace")
#plt.yticks(np.arange(0, 601, step=100))
#ax.set(xlabel="number of half spaces")
ax.legend(title="variables")
ax.set(ylabel="time")
fig.tight_layout()
fig.savefig(dir_name+instance+"Timeplott.jpg")


fig, ax = plt.subplots(figsize=(4,3))
dfy.groupby(['dimension','samplesize']).mean()["TPR:Mean"].unstack(level=0).plot(ax=ax,title=solver+instance+" Halfspace")
plt.yticks(np.arange(0.5, 1.05, step=0.1))
#ax.set(xlabel="number of half spaces")
ax.legend(title="variables")
ax.set(ylabel="TPR")
fig.tight_layout()

fig.savefig(dir_name+instance+"TPR-plott.jpg")

fig, ax = plt.subplots(figsize=(4,3))
dfy.groupby(['dimension','samplesize']).mean()['TNR:Mean'].unstack(level=0).plot(ax=ax,title=solver+instance+" Halfspace")
plt.yticks(np.arange(0.5, 1.05, step=0.1))
#ax.set(xlabel="number of half spaces")
ax.legend(title="variables")
ax.set(ylabel="TPR")
fig.tight_layout()
fig.savefig(dir_name+instance+"TNR-plott.jpg")