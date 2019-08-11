import pandas as pd
import os
with open("../data/processed/namelist.txt","r")as fin:
    names=eval(fin.read())
with open("../data/processed/kpi_list.txt","r")as fin:
    kpis=eval(fin.read())


for it in names:
    old=pd.read_csv("../data/inskpi_1s/"+it+"0407.csv",index_col="timestamp")
    new=pd.DataFrame(index=old.index,columns=kpis)
    for col in old.columns:
        if col in kpis:
            new[col]=old[col]
    new.to_csv("../data/processed/test_"+it+".csv")
    print(it)