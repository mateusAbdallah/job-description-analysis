# %%

import pandas as pd

# %%

df = pd.read_csv('../data/google_jobs_api.csv')
df.shape

# %% 

df.info(memory_usage='deep')
