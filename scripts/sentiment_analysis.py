# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

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
                      
df_concat.to_csv('../data/data_jobs.csv')
# %%

df_concat.info(memory_usage='deep')

# %%

df_concat.isnull().sum()

# %%

sentiments = df_concat['job_description'].apply(lambda x: TextBlob(x).sentiment)
sentiment_df = pd.DataFrame(sentiments.tolist(), columns=['polarity', 'subjectivity'])

# %%

plt.figure(figsize=(12,5))
sns.histplot(sentiment_df['polarity'], bins=30, kde=True, color='skyblue')
plt.title('Sentiment Polarity Distribution (Job Description)')
plt.xlabel('Polarity (-1: Negative, 1: Positive)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

