import pandas as pd
import os
with open("../data/processed/namelist.txt","r")as fin:
    names=eval(fin.read())
with open("../data/processed/kpi_list.txt","r")as fin:
    kpis=eval(fin.read())

df=None
for it in names:
    tdf=[]
    for jt in range(6):
        old=pd.read_csv("../data/inskpi_1s/"+it+"05"+str(jt+19)+".csv",index_col="timestamp")
        new=pd.DataFrame(index=old.index,columns=kpis)
        for col in old.columns:
            if col in kpis:
                new[col]=old[col]
        tdf.append(new.add_suffix(it))
    tdf=pd.concat(tdf)
    tdf.to_csv("../data/processed/train_"+it+".csv")
    print(it)