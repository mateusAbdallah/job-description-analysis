# %%

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


nltk.download('punkt')
nltk.download('stopwords')

# %%

df = pd.read_csv('../data/data_jobs.csv')

# %%

all_descriptions = ' '.join(df['job_description'].astype(str))

# %%

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english') + ['will', 'use', 'etc', 'including'])
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

cleaned_text = clean_text(all_descriptions)

# %%

word_freq = Counter(cleaned_text.split())

# %%

wordcloud = WordCloud(width=800, 
                      height=500, 
                      background_color='white',
                      max_words=100,
                      colormap='viridis').generate_from_frequencies(word_freq)
                    

# %%

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Data Science Job Descriptions', fontsize=16, pad=20)
plt.show()

# %%

print("\nTop 20 Words:")
for word, count in word_freq.most_common(20):
    print(f"{word}: {count}")