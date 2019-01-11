import  pandas as pd
import matplotlib.pyplot as plt

#dir_name = "../output/Observer exp/{}/{}.csv".format("cuben", "Results")
dir_name="/Users/Elias/Documents/GitHub/smtlearn/output/cuben/Cubenresults.csv"

data=pd.read_csv(dir_name,names=["run","dimension","samplesize","time","actual","synthesized","tpr","tnr"])

pname="cuben"

averageoverseed=data.groupby(["dimension","samplesize"]).mean()
std=data.groupby(["dimension","samplesize"]).std()

del averageoverseed["run"]
del std["run"]
averageoverseed["constrain_difference"]=averageoverseed["synthesized"]-averageoverseed["actual"]
std["constrain_difference"]=std["synthesized"]-std["actual"]

dir_name = "../output/{}/{}.csv".format(pname, "MergedMean")
averageoverseed.to_csv(dir_name)

dir_name = "../output/{}/{}.csv".format(pname, "MergedStd")
std.to_csv(dir_name)



#fig, ax = plt.subplots(figsize=(8,6))
#data.groupby(['dimension','samplesize']).mean()["time"].unstack(level=0).plot(ax=ax)