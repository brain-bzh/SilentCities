import pandas as pd
import sys

def remove_duplicates(Df):
    Df = Df.drop_duplicates(subset=['name', 'start', 'datetime'])
    return Df
csvfile = sys.argv[1]
Dftest = pd.read_csv(csvfile)

Dftest = remove_duplicates(Dftest)


Dftest.to_csv(csvfile, index=False)