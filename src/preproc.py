import pandas as pd
import os
import shutil
path="../data/inskpi_1s"
l=os.listdir(path)
df=pd.DataFrame(columns=["timestamp"]).set_index("timestamp")
i=0
name_dict=dict()
for name in l:
    i+=1
    name_dict[name]=i
for name in l:
    t=pd.read_csv(os.path.join(path,name),index_col="timestamp")
    df=df.join(t,how="outer",rsuffix=str(name_dict[name]))
    print(name)
df.fillna(0.0,inplace=True)
df.to_csv("../data/processed/a.csv")
