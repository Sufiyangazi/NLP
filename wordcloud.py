# %%
import pandas as pd
import matplotlib.pyplot as plt
# %%

# Load the dataset [android games]
file_name = 'C:\\Users\\Admin\\Downloads\\android-games.csv'
df = pd.read_csv(file_name)
df.head()
df.isna().sum()
# %%
df.head()
# %%
for cat in df.category:
    print(cat.split(' ')[1])
# %%
text = " ".join(cat.split()[1] for cat in df.category)
text
# %%
text = " ".join(cat.split()[1] for cat in df.category)
from wordcloud import WordCloud
Word_cloud = WordCloud(height=2000,width=2000,background_color='black')
Word_cloud = Word_cloud.generate(text)
plt.imshow(Word_cloud)
plt.title('Most common words in positive coustmr comment')
plt.axis('off')
plt.show()
# %%
