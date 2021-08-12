import sys
sys.path.insert(0, "../")
import pandas as pd

from visualisation import map 


database = pd.read_csv('/Users/nicolas/Documents/SilentCities/database/public_final_metadata_geo_stats.csv')


fig = map.get_map_fig(database)
fig.show()