import pandas as pd

df = pd.read_csv('label_and_questions.csv', index_col=0)

for i in range(len(df)):
    print(f'{i+1}.', df.loc[i, 'criterion'])
