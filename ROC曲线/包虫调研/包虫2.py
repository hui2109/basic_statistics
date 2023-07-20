import os.path

import numpy as np
import pandas as pd

df = pd.read_excel('./detail_answer.xlsx')
df = df.where(df.notnull(), None)
results = []

for i in range(len(df)):
    results.append(df.iloc[i, :].to_dict())

s = set()
for person in results:
    place = '_'.join(person['工作地址：'].split('-')[:-1])
    person['工作地址：'] = place
    s.add(place)

# district字典结构：
# key：城市
# value：list，每个list里装该城市的一些人，每个人是一个dict
district = {}
for place in s:
    district[place] = []

for key in district.keys():
    li = district[key]
    for person in results:
        if person['工作地址：'] == key:
            li.append(person)

for people in district.values():
    for person in people:  # person是一个字典
        for key, item in list(person.items()):
            if item is None:
                del person[key]

# analysis_dict
# key: city
# value: dict
# key: 题号
# value: list, 装各种答案
analysis_dict = {
    '四川_雅安市': dict(),
    '四川_阿坝藏族羌族自治州': dict(),
    '四川_成都市': dict(),
    '四川_甘孜藏族自治州': dict(),
    '四川_遂宁市': dict(),
    '四川_凉山彝族自治州': dict(),
    '四川_绵阳市': dict(),
    '西藏_拉萨市': dict()
}
for city, people in district.items():
    for person in people:  # person是一个字典
        for key, value in person.items():
            if '、' in key:
                if key not in analysis_dict[city].keys():
                    analysis_dict[city][key] = []
                    analysis_dict[city][key].append(value)
                else:
                    analysis_dict[city][key].append(value)

# 写入到文件
if os.path.exists('../../aresources/my_data/classify.xlsx'):
    os.remove('../../aresources/my_data/classify.xlsx')
df = pd.DataFrame()
with pd.ExcelWriter('../../aresources/my_data/classify.xlsx') as writer:
    df.to_excel(writer)

# 将字典的值转换为等长的列表
for city, questions in analysis_dict.items():
    max_length = max(len(v) for v in questions.values())
    a = {k: v + [np.nan] * (max_length - len(v)) for k, v in questions.items()}
    df = pd.DataFrame.from_dict(a)
    with pd.ExcelWriter('../../aresources/my_data/classify.xlsx', mode='a') as writer:
        df.to_excel(writer, sheet_name=city, index=False)
