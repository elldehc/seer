import pandas as pd
import os
path="../data/inskpi_1s"
l=os.listdir(path)
cl1=set()
cl2=set()
for name in l:
    print(name)
    t=set(pd.read_csv(os.path.join(path,name),index_col="timestamp").columns)
    if name[-8:-4]=="0407":
        if not t.issubset(cl1):
            print(t)
            cl1=cl1.union(t)
    else:
        if not t.issubset(cl2):
            print(t)
            cl2=cl2.union(t)

with open("../data/processed/kpi_list.txt","w")as fout:
    fout.write(str(cl1))
    

        
        
    