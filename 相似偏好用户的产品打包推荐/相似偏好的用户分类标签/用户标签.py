import pandas as pd
import numpy as np


def take_limit(df: pd.DataFrame, limit=5):
    result = pd.DataFrame({}, columns=df.columns)
    user_ids = df['用户号'].unique()
    for user_id in user_ids:
        user_types = df[df['用户号'] == user_id]
        result = result.append(user_types.head(limit))
    return result


columns = ['用户号', '产品名称', '推荐指数']

ejml = pd.read_csv('二级目录.csv', encoding='UTF-8', dtype=np.str)[columns]
ejml = take_limit(ejml, 4)
ejml['类型'] = '偏好二级目录'
flmc = pd.read_csv('分类名称.csv', encoding='UTF-8', dtype=np.str)[columns]
flmc = take_limit(flmc, 4)
flmc['类型'] = '偏好节目类型'
dq = pd.read_csv('地区.csv', encoding='UTF-8', dtype=np.str)[columns]
dq = take_limit(dq, 4)
dq['类型'] = '偏好地区'
yz = pd.read_csv('语种.csv', encoding='UTF-8', dtype=np.str)[columns]
yz = yz[yz[columns[1]] != '无']
yz = take_limit(yz, 4)
yz['类型'] = '偏好语种'

result = flmc.append(dq).append(yz)
result = result.reindex(columns=['用户号', '类型', '产品名称', '推荐指数'])

# result['三级标签'] = ''
result.columns = ['用户号', '一级标签', '二级标签', '推荐指数']
result.sort_values('用户号', ascending=True, inplace=True)
result.fillna('')
result.fillna('')

final = pd.DataFrame({}, columns=result.columns)
user_ids = result['用户号'].unique()
for user_id in user_ids:
    user_types = result[result['用户号'] == user_id].copy()
    # user_types = pd.DataFrame()
    user_types.sort_values('推荐指数', ascending=False, inplace=True)
    final = final.append(user_types)

result = final

result.to_csv('问题二推荐结果表.csv', encoding='UTF-8', index=False)
result.to_csv('问题二推荐结果表_GBK.csv', encoding='GBK', index=False)
df = pd.read_csv('用户基本信息.csv', encoding='GBK', dtype=np.str)
result = result.merge(df, on='用户号', how='left')
result.to_csv('用户数据标签表.csv', encoding='UTF-8', index=False)
result.to_csv('用户数据标签表_GBK.csv', encoding='GBK', index=False)

# 用户标签体系表
result = result[['一级标签', '二级标签']].drop_duplicates()
# result = pd.DataFrame()
result.to_csv('用户标签体系表.csv', encoding='UTF-8', index=False)
result.to_csv('用户标签体系表_GBK.csv', encoding='GBK', index=False)
