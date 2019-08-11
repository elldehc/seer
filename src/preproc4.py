import os
path="../data/inskpi_1s"
l=os.listdir(path)
s=set()
for it in l:
    s.add(it[:-8])
with open("../data/processed/namelist.txt","w")as fout:
    fout.write(str(s))