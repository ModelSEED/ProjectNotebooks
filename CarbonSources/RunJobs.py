import sys
import os
index = int(sys.argv[1])
count = int(sys.argv[2])
lines = []
with open('GenomeList.txt') as f:
    lines = f.readlines()
iterator = 0
for line in lines:
    mod = iterator % count 
    if mod == index:
        line = line. rstrip("\n")
        array = line.split("\t")
        os.system("python GapfillGenomeModels.py "+" ".join(array))
    iterator += 1