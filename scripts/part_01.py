# %%

import pandas as pd

# %%

df_da = pd.read_csv('../data/data_analyst.csv')
df_da.head()

# %%

df_ds = pd.read_csv('../data/data_scientist.csv')
df_ds.head()

# %% 

columns_dropped_da = ['Unnamed: 0', 'Salary Estimate',
       'Rating', 'Company Name', 'Headquarters', 'Size', 'Founded',
       'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Competitors',
       'Easy Apply'
       ]

df_da = df_da.drop(columns=columns_dropped_da)

# %%

columns_dropped_ds = ['Unnamed: 0', 'index', 'Salary Estimate',
        'Rating', 'Company Name',  'Headquarters',
       'Size', 'Founded', 'Type of ownership', 'Industry', 'Sector', 'Revenue',
       'Competitors', 'Easy Apply'
       ]


df_ds = df_ds.drop(columns=columns_dropped_ds)

# %%

df_concat = pd.concat([df_da, df_ds])

# %%

df_concat = (df_concat.rename(columns={'Job Title': 'job_title', 
                                      'Job Description': 'job_description', 
                                      'Location': 'location'})
                       .drop_duplicates())
                      

# %%


