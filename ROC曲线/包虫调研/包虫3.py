import pandas as pd

df1 = pd.read_excel('./classify.xlsx', sheet_name='四川_阿坝藏族羌族自治州')
df2 = pd.read_excel('./classify.xlsx', sheet_name='四川_甘孜藏族自治州')
df3 = pd.read_excel('./classify.xlsx', sheet_name='四川_凉山彝族自治州')

df = pd.concat([df1, df2, df3])
# print(df)

with pd.ExcelWriter('../../aresources/my_data/classify.xlsx', mode='a') as writer:
    df.to_excel(writer, index=False, sheet_name='阿坝_甘孜_凉山')
