import pandas as pd
import os
import shutil
path="../data/inskpi_1s"
l=set(os.listdir(path))
nl=["0407","0519","0520","0521","0522","0523","0524"]
for name in l:
    if not(name[:-8]+"0407.csv" in l):
        #print(name,"0407")
        for it in nl:
            if name[:-8]+it+".csv" in l:
                try:
                    os.remove(os.path.join(path,name[:-8]+it+".csv"))
                except FileNotFoundError:
                    pass
    