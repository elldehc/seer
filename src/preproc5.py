import pandas as pd
import os

with open("../data/processed/namelist.txt","r")as fin:
    names=eval(fin.read())
l=[]

for it in names:
    df=pd.read_csv("../data/inskpi_1s/"+it+"0407.csv",index_col="timestamp")
    if "tcp_rt" in df.columns:
        l.extend(df["tcp_rt"])
        for j in range(19,25):
            df=pd.read_csv("../data/inskpi_1s/"+it+"05"+str(j)+".csv",index_col="timestamp")
            l.extend(df["tcp_rt"])
    print(it)
l.sort()
thr=l[int(0.99*len(l))]
print("ok")

ans=pd.DataFrame(data=0,index=range(1554566400,1554566400+86400,5),columns=["ans","tcp_rt"])
ans2=pd.DataFrame(data=0,index=range(1558195200,1558195200+86400*6,5),columns=["ans","tcp_rt"])
x=0
for it in names:
    x+=1
    df=pd.read_csv("../data/inskpi_1s/"+it+"0407.csv",index_col="timestamp")
    if "tcp_rt" in df.columns:
        for i in df.index:
            if df["tcp_rt"][i]>thr and df["tcp_rt"][i]>ans["tcp_rt"][i]:
                ans["tcp_rt"][i]=df["tcp_rt"][i]
                ans["ans"][i]=x
        for j in range(19,25):
            df=pd.read_csv("../data/inskpi_1s/"+it+"05"+str(j)+".csv",index_col="timestamp")
            for i in df.index:
                if df["tcp_rt"][i]>thr and df["tcp_rt"][i]>ans2["tcp_rt"][i]:
                    ans2["tcp_rt"][i]=df["tcp_rt"][i]
                    ans2["ans"][i]=x
    print(it)
            
ans.rename_axis("timestamp").drop("tcp_rt",axis=1).to_csv("../data/processed/test_ans.csv")
ans2.rename_axis("timestamp").drop("tcp_rt",axis=1).to_csv("../data/processed/train_ans.csv")
