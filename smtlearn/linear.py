import  pandas as pd
import matplotlib.pyplot as plt

dir_name = "../output/Observer exp/{}/{}.csv".format("cuben", "Results")
data=pd.read_csv(dir_name,names=["run","dimension","samplesize","time","actual","synthesized","accuracy"])



averageoverseed=data.groupby(["samplesize","dimension"], as_index=False).mean()

del averageoverseed["run"]
averageoverseed["constrain_difference"]=averageoverseed["synthesized"]-averageoverseed["actual"]


averageoverseed["notime"]=(averageoverseed["time"]-averageoverseed["time"].min())/(averageoverseed["time"].max()-averageoverseed["time"].min())
averageoverseed["noaccuracy"]=(averageoverseed["accuracy"]-averageoverseed["accuracy"].min())/(averageoverseed["accuracy"].max()-averageoverseed["accuracy"].min())

averageoverseed["objective"]=averageoverseed["noaccuracy"]-averageoverseed["notime"]
averageoverseed["percent"]=averageoverseed["dimension"]/averageoverseed["samplesize"]

max=averageoverseed.groupby(["samplesize"],as_index=False)["objective"].idxmax()

ob=averageoverseed.loc[max]



ob["percent"]=ob["dimension"]/ob["samplesize"]
l=ob[["samplesize","dimension","percent","accuracy"]]

ob.plot(x="samplesize",y="percent")

dir_name = "../output/Observer exp/{}/{}.csv".format("cuben", "MergedMean")
averageoverseed.to_csv(dir_name)