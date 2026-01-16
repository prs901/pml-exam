import pandas


df = pandas.read_csv('results_clean.csv')

df.loc[:, 'train_time'] /= 60
df.loc[df['model'].isin(('ae', 'fm')), 'gen_time'] *= 1000

sorter = ['ddpm', 'ddim', 'fm', 'ae']
df.sort_values(by="model", key=lambda column: column.map(lambda e: sorter.index(e)), inplace=True)

df_model = df.groupby('model', sort=False)

means = df_model.mean()
stds = df_model.std(ddof=0)

print(f'\n\nDATA\n{"-"*20}')
print(df)

print(f'\n\nMEANS\n{"-"*20}')
print(means)

print(f'\n\nSTDS\n{"-"*20}')
print(stds)
