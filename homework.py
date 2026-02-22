import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path="iris.csv"):
	if os.path.exists(path):
		try:
			return pd.read_csv(path)
		except Exception as e:
			print(f"Failed to read '{path}': {e}")
	print("Using seaborn's 'iris' dataset as fallback.")
	return sns.load_dataset('iris')
df = load_data()

# normalize column names for convenience
original_cols = df.columns.tolist()
df.columns = [c.strip().lower().replace(' ', '_') for c in original_cols]
print(df.info())

# helper to get column by candidate names

def find_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

species = find_col(['species'])
sepal_len = find_col(['sepallengthcm','sepal_length','sepal_length_cm'])
sepal_wid = find_col(['sepalwidthcm','sepal_width','sepal_width_cm'])

if species and sepal_len:
    sns.barplot(x=species, y=sepal_len, data=df)
    plt.title('Mean sepal length by species')
    plt.show()

if species:
    sns.countplot(x=species, data=df)
    plt.title('Count by species')
    plt.show()

if species and sepal_wid:
    sns.boxplot(x=species, y=sepal_wid, data=df)
    plt.title('Sepal width distribution by species')
    plt.show()
    sns.swarmplot(x=species, y=sepal_wid, data=df, color='k', alpha=0.6)
    plt.title('Sepal width swarm by species')
    plt.show()

if sepal_wid:
    sns.histplot(df[sepal_wid], kde=True)
    plt.title('Sepal width distribution')
    plt.show()

if sepal_len and sepal_wid:
    sns.jointplot(x=sepal_len, y=sepal_wid, data=df, kind='scatter')
    plt.show()

num = df.select_dtypes(include='number')
if not num.empty:
    sns.pairplot(num)
    plt.show()