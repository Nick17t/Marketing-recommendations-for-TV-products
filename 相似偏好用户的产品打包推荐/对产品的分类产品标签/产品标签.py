# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import time

start_time = time.clock()
print('启动于：{}'.format(start_time))

df = pd.read_csv('tv_programme.csv', encoding='UTF-8')
yhjm = pd.read_csv('for_analysis_time.csv', encoding='GBK')
yhjm = yhjm[['节目名称', '二级目录']]
df = df.merge(yhjm, 'left', left_on='正题名', right_on='节目名称')
final = pd.DataFrame({}, columns=['一级标签', '二级标签'])
tmp = df[['分类名称']]
tmp.drop_duplicates(inplace=True)
tmp.columns = ['二级标签']
tmp['一级标签'] = '节目分类'
tmp = tmp.reindex(columns=['一级标签', '二级标签'])
final = final.append(tmp)

tmp = df[['字母语种']]
tmp.drop_duplicates(inplace=True)
tmp.columns = ['二级标签']
tmp['一级标签'] = '语言'
tmp = tmp.reindex(columns=['一级标签', '二级标签'])
final = final.append(tmp)

tmp = df[['地区参数']]
tmp.drop_duplicates(inplace=True)
tmp.columns = ['二级标签']
tmp['一级标签'] = '地区'
tmp = tmp.reindex(columns=['一级标签', '二级标签'])
final = final.append(tmp)

tmp = df[['二级目录']]
tmp.drop_duplicates(inplace=True)
tmp = tmp[tmp['二级目录'].notna() & tmp['二级目录'].notnull()]
tmp.columns = ['二级标签']
tmp['一级标签'] = '节目归属'
tmp = tmp.reindex(columns=['一级标签', '二级标签'])
final = final.append(tmp)

final.to_csv('产品标签体系表.csv', encoding='UTF-8', index=False)
final.to_csv('产品标签体系表_GBK.csv', encoding='GBK', index=False)

# 产品数据标签

final = df
final['一级标签'] = '分类名称'
final['二级标签'] = final['分类名称']
final.drop(columns=yhjm.columns, inplace=True)
final.to_csv('产品数据标签.csv', encoding='UTF-8', index=False)
final.to_csv('产品数据标签_GBK.csv', encoding='GBK', index=False)

# 用户收视数据标签

df = pd.read_csv('../tv_programme.csv', encoding='UTF-8', dtype=np.str)
df = df[['正题名', '字母语种', '地区参数']]
df.drop_duplicates(inplace=True)
yhjm = pd.read_csv('../for_analysis_time.csv', encoding='GBK', dtype=np.str)
simpler = np.random.permutation(len(yhjm))
yhjm = yhjm.take(simpler)  # 打乱数据
yhjm = yhjm[0:200]  # 取一千条数据
# yhjm = yhjm[['节目名称', '二级目录']]
df = yhjm.merge(df, 'right', right_on='正题名', left_on='节目名称')
df = df[df['节目名称'].notna()]
df = df.reset_index(drop=True)

final = pd.DataFrame({}, columns=yhjm.columns)
for index in df.index:
    row = df.loc[index]
    row1 = row.copy()
    row1['一级标签'] = '节目分类'
    row1['二级标签'] = row1['分类名称']
    final = final.append(row1)
    row1 = row.copy()
    row1['一级标签'] = '地区'
    row1['二级标签'] = row1['地区参数']
    final = final.append(row1)
    row1 = row.copy()
    row1['一级标签'] = '节目归属'
    row1['二级标签'] = row1['二级目录']
    final = final.append(row1)
    row1 = row.copy()
    row1['一级标签'] = '语言'
    row1['二级标签'] = row1['字母语种']
    final = final.append(row1)

final.to_csv('用户收视数据标签.csv', encoding='UTF-8', index=False)
final.to_csv('用户收视数据标签_GBK.csv', encoding='GBK', index=False)

end_time = time.clock()
duration = end_time - start_time
print('结束于：{}，耗时：{}'.format(end_time, duration))
