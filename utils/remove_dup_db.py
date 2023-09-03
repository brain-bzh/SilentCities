import pandas as pd
import sys

def remove_duplicates(Df):
    Df = Df.drop_duplicates(subset=['datetime'])
    return Df
csvfile = sys.argv[1]
Dftest = pd.read_csv(csvfile)

Dftest_rem = remove_duplicates(Dftest)
doremove = True
if len(Dftest) != len(Dftest_rem):
    print(f"Removing duplicate lines in {csvfile}")
    print(len(Dftest),len(Dftest_rem))
    print("-----")
    if doremove:
        print("overwriting file")
        Dftest_rem.to_csv(csvfile, index=False)