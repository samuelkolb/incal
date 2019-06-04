

import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




def postporcessresults(solver,instance,importfile,save=1):

    dir_name = "/Users/Elias/Documents/GitHub/smtlearn/output/{}/{}/".format(solver,instance)

    data=pd.read_csv(dir_name+importfile,index_col=0,names=["dimension","samplesize","time","actual","synthesized","tpr","tnr"])

    
    if data["time"].dtypes=='object':
        data['time'] = pd.to_numeric(data['time'],errors='coerce')
        data["time"]=data["time"].fillna(5401)
    
    data = data[data.samplesize != 400]
    data.loc[data.time > 5400,"time"] = 5400
    datafortime=data.copy()
    data = data[data["time"] <= 5399]
    
    averageoverseed=data.groupby(["dimension","samplesize"]).mean()
    averageoverseed["timeEX"]=averageoverseed["time"]
    
    dataforstd=datafortime.groupby(["dimension","samplesize"]).mean()
    averageoverseed["time"]=dataforstd["time"]
    
    stdgroup=data.groupby(["dimension","samplesize"])

    
    dataforstd=datafortime.groupby(["dimension","samplesize"])

    stdtime=[]
    for [label, df] in dataforstd:
        timestd=np.std(df["time"], axis=0) / np.sqrt(df["time"].shape[0])
        stdtime.append(timestd)

    
    stdtime=np.array(stdtime)
    
    
    stdrest=[]
    for [label, df] in stdgroup:
        timestd=np.std(df["time"], axis=0) / np.sqrt(df["time"].shape[0])
        tprstd=np.std(df["tpr"], axis=0) / np.sqrt(df["tpr"].shape[0])
        tnrstd=np.std(df["tnr"], axis=0) / np.sqrt(df["tnr"].shape[0])
        
        stdrest.append([label[0],label[1],timestd,tprstd,tnrstd])
    
    std=data.groupby(["dimension","samplesize"]).std()
    std["Constrain_difference:Std"]=std["synthesized"]-std["actual"]

    stdtnrtpr=pd.DataFrame(stdrest,columns=["s","d",'time', 'tpr', 'tnr'])
    std=std.reset_index()
    
    std["timeEX"]=stdtnrtpr["time"]
    std["time"]=stdtime#stdk["time"]
    std["tpr"]=stdtnrtpr["tpr"]
    std["tnr"]=stdtnrtpr["tnr"]

    std=std.set_index(['dimension', 'samplesize'])

    averageoverseed["Constrain_difference:Mean"]=averageoverseed["synthesized"]-averageoverseed["actual"]

    averageoverseed.columns=["Time:Mean","ActualD","Found:mean","TPR:Mean","TNR:Mean","timeEXmean","Constrain_difference:Mean"]
    std.columns=["Time:STD","Actual","Found:STD","TPR:STD","TNR:STD","Constrain_difference:Std","timeEXstd"]
    
    meanandstd=pd.concat([averageoverseed, std],axis=1)
    
    del meanandstd["ActualD"]
    del meanandstd["Found:mean"]
    del meanandstd["Found:STD"]
    
    meanandstd=meanandstd[['Time:Mean','Time:STD',"Constrain_difference:Mean" ,"Constrain_difference:Std", 'TPR:Mean', 'TPR:STD', 'TNR:Mean', 'TNR:STD',"timeEXmean","timeEXstd"]]
    
    colsto1 = ['Time:Mean','Time:STD']
    meanandstd[colsto1] = meanandstd[colsto1].round(1)
    colsto2 = ['TPR:Mean', 'TPR:STD', 'TNR:Mean', 'TNR:STD']
    meanandstd[colsto2] = meanandstd[colsto2].round(2)
    
    if save==0:
        meanandstd.to_csv(dir_name+instance+"overallrunsresults.csv")

    return meanandstd.reset_index()




    
def geometricproblems(data1, data2, data3, yvalue, ystd, ylabel, line1label, line2label, line3label, plotname, scale=1):

    dir_name = "/Users/Elias/Desktop/"
    fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(12,5),sharey='row')


    fig.suptitle("Cuben")
    c=1
    for [label, df], axx in zip(data1.groupby('dimension'),ax):
        labellegend= line1label
        df.plot(x="samplesize",y=yvalue,ax=axx,label=labellegend,color="#1f77b4")
        axx.fill_between(df["samplesize"], df[yvalue] - df[ystd], df[yvalue] + df[ystd], color="#1f77b4",alpha=0.35, linewidth=0)
        
    for [label, df], axx in zip(data3.groupby('dimension'),ax):
        labellegend= line3label
        df.plot(x="samplesize",y=yvalue,ax=axx,label=labellegend,color="#2ca02c")
        axx.fill_between(df["samplesize"], df[yvalue] - df[ystd], df[yvalue] + df[ystd], color="#2ca02c",alpha=0.35, linewidth=0)

    for [label, df], axx in zip(data2.groupby('dimension'),ax):
        labellegend=line2label
        df.plot(x="samplesize",y=yvalue,ax=axx,label=labellegend,color="#ff7f0e")

        axx.fill_between(df["samplesize"], df[yvalue] - df[ystd], df[yvalue] + df[ystd], color="#ff7f0e",alpha=0.35, linewidth=0)
       
        c=c+1
        if scale==1:
            axx.set_ylim([0.7,1])

        axx.set(xlabel="Sample size")
        axx.set(ylabel=ylabel)
        axx.set(title="n: "+str(label))

    
    #labels=["IncaLP(MILP)","MILPCS","IncaLP(SMT)"]
    #ax[0].legend(labels)
    
    ax[1].get_legend().remove()
    ax[0].get_legend().remove()
    ax[2].get_legend().remove()
    ax[3].get_legend().remove()

    
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.savefig(dir_name+plotname+".png",bbox_inches='tight', pad_inches=0)
    
    
def textbookproblems(name,data1, data2=None, data3=None, data4=None):
    
    dir_name = "/Users/Elias/Desktop/"
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(12,5))
    
    fig.suptitle(name)


    data1.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#1f77b4")
    ax[0].fill_between(data1["samplesize"], data1["Time:Mean"] - data1["Time:STD"], data1["Time:Mean"] + data1["Time:STD"], color="#1f77b4",alpha=0.35, linewidth=0)
    
    data2.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#ff7f0e")
    ax[0].fill_between(data2["samplesize"], data2["Time:Mean"] - data2["Time:STD"], data2["Time:Mean"] + data2["Time:STD"], color="#ff7f0e",alpha=0.35, linewidth=0)   
    
    data3.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#2ca02c")
    ax[0].fill_between(data3["samplesize"], data3["Time:Mean"] - data3["Time:STD"], data3["Time:Mean"] + data3["Time:STD"], color="#2ca02c",alpha=0.35, linewidth=0)

    data4.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#8B008B")
    ax[0].fill_between(data4["samplesize"], data4["Time:Mean"] - data4["Time:STD"], data4["Time:Mean"] + data4["Time:STD"], color="#8B008B",alpha=0.35, linewidth=0)
    

    data1.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#1f77b4")
    ax[1].fill_between(data1["samplesize"], data1["TPR:Mean"] - data1["TPR:STD"], data1["TPR:Mean"] + data1["TPR:STD"], color="#1f77b4",alpha=0.35, linewidth=0)
    
    data2.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#ff7f0e")
    ax[1].fill_between(data2["samplesize"], data2["TPR:Mean"] - data2["TPR:STD"], data2["TPR:Mean"] + data2["TPR:STD"], color="#ff7f0e",alpha=0.35, linewidth=0)   
    
    data3.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#2ca02c")
    ax[1].fill_between(data3["samplesize"], data3["TPR:Mean"] - data3["TPR:STD"], data3["TPR:Mean"] + data3["TPR:STD"], color="#2ca02c",alpha=0.35, linewidth=0)

    data4.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#8B008B")
    ax[1].fill_between(data4["samplesize"], data4["TPR:Mean"] - data4["TPR:STD"], data4["TPR:Mean"] + data4["TPR:STD"], color="#8B008B",alpha=0.35, linewidth=0)

    data1.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#1f77b4")
    ax[2].fill_between(data1["samplesize"], data1["TNR:Mean"] - data1["TNR:STD"], data1["TNR:Mean"] + data1["TNR:STD"], color="#1f77b4",alpha=0.35, linewidth=0)
    
    data2.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#ff7f0e")
    ax[2].fill_between(data2["samplesize"], data2["TNR:Mean"] - data2["TNR:STD"], data2["TNR:Mean"] + data2["TNR:STD"], color="#ff7f0e",alpha=0.35, linewidth=0)   
    
    data3.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#2ca02c")
    ax[2].fill_between(data3["samplesize"], data3["TNR:Mean"] - data3["TNR:STD"], data3["TNR:Mean"] + data3["TNR:STD"], color="#2ca02c",alpha=0.35, linewidth=0)
     
    data4.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#8B008B")
    ax[2].fill_between(data4["samplesize"], data4["TNR:Mean"] - data4["TNR:STD"], data4["TNR:Mean"] + data4["TNR:STD"], color="#8B008B",alpha=0.35, linewidth=0)

    ax[1].set_ylim([0.7,1])
    ax[2].set_ylim([0.7,1])


    ax[0].get_legend().remove()
    ax[1].get_legend().remove()
    ax[2].get_legend().remove()

    labels=["IncaLP(MILP)","IncaLP(SMT)","MILPCS","IncaLP(SMT) \n Extended"]
    ax[2].legend(labels)

    ax[0].set(ylabel="Time (s)")
    ax[1].set(ylabel="True positive rate")
    ax[2].set(ylabel="True negative rate")
    
#    ax[0].set(xlabel="Sample size")
#    ax[1].set(xlabel="Sample size")
#    ax[2].set(xlabel="Sample size")
    
    ax[0].set(xlabel="Initial active set")
    ax[1].set(xlabel="Initial active set")
    ax[2].set(xlabel="Initial active set")
    
    ax[0].set_title("Time (s)")
    ax[1].set_title("True positive rate")
    ax[2].set_title("True negative rate")
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    fig.savefig(dir_name+name+".png",bbox_inches='tight', pad_inches=0)
    
    
    
def selctionandper_m(data1, data2, data3, data4, name):
    
    dir_name = "/Users/Elias/Desktop/"
    plotname=name
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(12,5))
    
    fig.suptitle(plotname)

#for H use FF0000 for milp cosnttnat 606060 #606060 for data 2 and DT EXP

    data1.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#1f77b4")
    ax[0].fill_between(data1["samplesize"], data1["Time:Mean"] - data1["Time:STD"], data1["Time:Mean"] + data1["Time:STD"], color="#1f77b4",alpha=0.35, linewidth=0)
    
    data2.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#FF0000")
    ax[0].fill_between(data2["samplesize"], data2["Time:Mean"] - data2["Time:STD"], data2["Time:Mean"] + data2["Time:STD"], color="#FF0000",alpha=0.35, linewidth=0)   

    data3.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#ff7f0e")
    ax[0].fill_between(data3["samplesize"], data3["Time:Mean"] - data3["Time:STD"], data3["Time:Mean"] + data3["Time:STD"], color="#ff7f0e",alpha=0.35, linewidth=0) 
     
    data4.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#00CCCC")
    ax[0].fill_between(data3["samplesize"], data4["Time:Mean"] - data4["Time:STD"], data4["Time:Mean"] + data4["Time:STD"], color="#00CCCC",alpha=0.35, linewidth=0)
    
    data1.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#1f77b4")
    ax[1].fill_between(data1["samplesize"], data1["TPR:Mean"] - data1["TPR:STD"], data1["TPR:Mean"] + data1["TPR:STD"], color="#1f77b4",alpha=0.35, linewidth=0)
    
    data2.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#FF0000")
    ax[1].fill_between(data2["samplesize"], data2["TPR:Mean"] - data2["TPR:STD"], data2["TPR:Mean"] + data2["TPR:STD"],color="#FF0000",alpha=0.35, linewidth=0)   
    
    data3.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#ff7f0e")
    ax[1].fill_between(data3["samplesize"], data3["TPR:Mean"] - data3["TPR:STD"], data3["TPR:Mean"] + data3["TPR:STD"],color="#ff7f0e",alpha=0.35, linewidth=0) 

    data4.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#00CCCC")
    ax[1].fill_between(data4["samplesize"], data4["TPR:Mean"] - data4["TPR:STD"], data4["TPR:Mean"] + data4["TPR:STD"],color="#00CCCC",alpha=0.35, linewidth=0)

    data1.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#1f77b4")
    ax[2].fill_between(data1["samplesize"], data1["TNR:Mean"] - data1["TNR:STD"], data1["TNR:Mean"] + data1["TNR:STD"], color="#1f77b4",alpha=0.35, linewidth=0)
    
    data2.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#FF0000")
    ax[2].fill_between(data2["samplesize"], data2["TNR:Mean"] - data2["TNR:STD"], data2["TNR:Mean"] + data2["TNR:STD"], color="#FF0000",alpha=0.35, linewidth=0)   

    data3.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#ff7f0e")
    ax[2].fill_between(data3["samplesize"], data3["TNR:Mean"] - data3["TNR:STD"], data3["TNR:Mean"] + data3["TNR:STD"], color="#ff7f0e",alpha=0.35, linewidth=0)     

    data4.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#00CCCC")
    ax[2].fill_between(data4["samplesize"], data4["TNR:Mean"] - data4["TNR:STD"], data4["TNR:Mean"] + data4["TNR:STD"], color="#00CCCC",alpha=0.35, linewidth=0)

    ax[1].set_ylim([0.7,1])
    ax[2].set_ylim([0.7,1])


    ax[0].get_legend().remove()
    ax[1].get_legend().remove()

    #labels=["SelectDT IncaLP(MILP)","Random selection IncaLP(MILP)","SelectDT IncaLP(SMT)","Random selection IncaLP(SMT)"]
    labels=["IncaLP(MILP)","IncaLP(SMT)","IncaLP(MILP) \n constant OF"]
    ax[2].legend(labels)
    
    ax[0].axvline(x=3, color='r', linestyle='-')
    ax[1].axvline(x=3, color='r', linestyle='-')
    ax[2].axvline(x=3, color='r', linestyle='-')
    ax[0].set_xticks([2, 4, 8, 16, 32, 64])
    ax[1].set_xticks([2, 4, 8, 16, 32, 64])
    ax[2].set_xticks([2, 4, 8, 16, 32, 64])

    ax[0].set(ylabel="Time (s)")
    ax[1].set(ylabel="True positive rate")
    ax[2].set(ylabel="True negative rate")
    
# =============================================================================
#     ax[0].set(xlabel="Sample size")
#     ax[1].set(xlabel="Sample size")
#     ax[2].set(xlabel="Sample size")
#     
# =============================================================================
    ax[0].set(xlabel="m")
    ax[1].set(xlabel="m")
    ax[2].set(xlabel="m")
    
# =============================================================================
#     ax[0].set_title("Time (s)")
#     ax[1].set_title("True positive rate")
#     ax[2].set_title("True negative rate")
# =============================================================================
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    
    fig.savefig(dir_name+plotname+".png",bbox_inches='tight', pad_inches=0)
    
    
    
    
def geometric(data1, data2=None, data3=None, data4=None, data5=None, data6=None):
    
    dir_name = "/Users/Elias/Desktop/"
    plotname="Pollution"
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(12,5))
    
    fig.suptitle(plotname)


    data1.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#1f77b4")
    ax[0].fill_between(data1["samplesize"], data1["Time:Mean"] - data1["Time:STD"], data1["Time:Mean"] + data1["Time:STD"], color="#1f77b4",alpha=0.35, linewidth=0)
    
    data2.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#ff7f0e")
    ax[0].fill_between(data2["samplesize"], data2["Time:Mean"] - data2["Time:STD"], data2["Time:Mean"] + data2["Time:STD"], color="#ff7f0e",alpha=0.35, linewidth=0)   
    
    data3.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#2ca02c")
    ax[0].fill_between(data3["samplesize"], data3["Time:Mean"] - data3["Time:STD"], data3["Time:Mean"] + data3["Time:STD"], color="#2ca02c",alpha=0.35, linewidth=0)

    data4.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#8B008B")
    ax[0].fill_between(data4["samplesize"], data4["Time:Mean"] - data4["Time:STD"], data4["Time:Mean"] + data4["Time:STD"], color="#8B008B",alpha=0.35, linewidth=0)
    
    data5.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#FF0000")
    ax[0].fill_between(data5["samplesize"], data5["Time:Mean"] - data5["Time:STD"], data5["Time:Mean"] + data5["Time:STD"], color="#FF0000",alpha=0.35, linewidth=0)
    
    data6.plot(x="samplesize",y="Time:Mean",ax=ax[0],color="#FF00FF")
    ax[0].fill_between(data6["samplesize"], data6["Time:Mean"] - data6["Time:STD"], data6["Time:Mean"] + data6["Time:STD"], color="#FF00FF",alpha=0.35, linewidth=0)
    
    data1.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#1f77b4")
    ax[1].fill_between(data1["samplesize"], data1["TPR:Mean"] - data1["TPR:STD"], data1["TPR:Mean"] + data1["TPR:STD"], color="#1f77b4",alpha=0.35, linewidth=0)
    
    data2.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#ff7f0e")
    ax[1].fill_between(data2["samplesize"], data2["TPR:Mean"] - data2["TPR:STD"], data2["TPR:Mean"] + data2["TPR:STD"], color="#ff7f0e",alpha=0.35, linewidth=0)   
    
    data3.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#2ca02c")
    ax[1].fill_between(data3["samplesize"], data3["TPR:Mean"] - data3["TPR:STD"], data3["TPR:Mean"] + data3["TPR:STD"], color="#2ca02c",alpha=0.35, linewidth=0)

    data4.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#8B008B")
    ax[1].fill_between(data4["samplesize"], data4["TPR:Mean"] - data4["TPR:STD"], data4["TPR:Mean"] + data4["TPR:STD"], color="#8B008B",alpha=0.35, linewidth=0)
  
    data5.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#FF0000")
    ax[1].fill_between(data5["samplesize"], data5["TPR:Mean"] - data5["TPR:STD"], data5["TPR:Mean"] + data5["TPR:STD"], color="#FF0000",alpha=0.35, linewidth=0)
    
    data6.plot(x="samplesize",y="TPR:Mean",ax=ax[1],color="#FF00FF")
    ax[1].fill_between(data6["samplesize"], data6["TPR:Mean"] - data6["TPR:STD"], data6["TPR:Mean"] + data6["TPR:STD"], color="#FF00FF",alpha=0.35, linewidth=0)

    data1.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#1f77b4")
    ax[2].fill_between(data1["samplesize"], data1["TNR:Mean"] - data1["TNR:STD"], data1["TNR:Mean"] + data1["TNR:STD"], color="#1f77b4",alpha=0.35, linewidth=0)
    
    data2.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#ff7f0e")
    ax[2].fill_between(data2["samplesize"], data2["TNR:Mean"] - data2["TNR:STD"], data2["TNR:Mean"] + data2["TNR:STD"], color="#ff7f0e",alpha=0.35, linewidth=0)   
    
    data3.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#2ca02c")
    ax[2].fill_between(data3["samplesize"], data3["TNR:Mean"] - data3["TNR:STD"], data3["TNR:Mean"] + data3["TNR:STD"], color="#2ca02c",alpha=0.35, linewidth=0)
     
    data4.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#8B008B")
    ax[2].fill_between(data4["samplesize"], data4["TNR:Mean"] - data4["TNR:STD"], data4["TNR:Mean"] + data4["TNR:STD"], color="#8B008B",alpha=0.35, linewidth=0)
    
    data5.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#FF0000")
    ax[2].fill_between(data5["samplesize"], data5["TNR:Mean"] - data5["TNR:STD"], data5["TNR:Mean"] + data5["TNR:STD"], color="#FF0000",alpha=0.35, linewidth=0)

    data6.plot(x="samplesize",y="TNR:Mean",ax=ax[2],color="#FF00FF")
    ax[2].fill_between(data6["samplesize"], data6["TNR:Mean"] - data6["TNR:STD"], data6["TNR:Mean"] + data6["TNR:STD"], color="#FF00FF",alpha=0.35, linewidth=0)



    ax[1].set_ylim([0.7,1])
    ax[2].set_ylim([0.7,1])


    ax[0].get_legend().remove()
    ax[1].get_legend().remove()


    labels=["IncaLP(MILP)","IncaLP(SMT)","MILPCS","IncaLP(SMT) \n Extended","IncaLP(MILP) \n constant objective","MILPCS reduced objective"]
    ax[2].legend(labels)

    ax[0].set(ylabel="Time (s)")
    #ax[1][0].set(ylabel="Time (s)")
    ax[1].set(ylabel="True positive rate")
    ax[2].set(ylabel="True negative rate")
    
    ax[0].set(xlabel="Sample size")
    ax[1].set(xlabel="Sample size")
    ax[2].set(xlabel="Sample size")
    
# =============================================================================
#     ax[0].set(xlabel="Initial active set")
#     ax[1].set(xlabel="Initial active set")
#     ax[2].set(xlabel="Initial active set")
# =============================================================================
    
    ax[0].set_title("Time (s)")
    ax[1].set_title("True positive rate")
    ax[2].set_title("True negative rate")
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    fig.savefig(dir_name+plotname+".png",bbox_inches='tight', pad_inches=0)





