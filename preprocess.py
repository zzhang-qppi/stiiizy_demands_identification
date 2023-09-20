import pandas
import os

import pandas as pd

file_names = os.listdir('raw_data')

target = 'yelp'
ihj = [n if target in n else None for n in file_names]
all_files = []
length = 0

for name in ihj:
    if name is None:
        continue
    df = pd.read_excel(os.path.join('raw_data', name)).loc[:, ['css-chan6m','raw__09f24__T4Ezm']]
    df.columns = ['date','comment']
    length += len(df)
    l = len(df)
    df.dropna(subset='comment', inplace=True)
    print(len(df)-l)
    all_files.append(df)

all_data = pd.concat(all_files)
all_data.reset_index(drop=True, inplace=True)
all_data.to_csv(f'data/{target}.csv')
print(length)
print(len(all_data))
