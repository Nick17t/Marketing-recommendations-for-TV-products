# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import time
import Recommender

from sqlalchemy import create_engine


class TVRecommend:
    output_dir = './result/'
    data = None
    cleaned = None
    score_matrix = None
    interest_matrix = None
    train_matrix = None
    recommender = Recommender.Recommender()
    limit = False
    recommend_count = None
    column_name = ('用户号', '节目名称')
    time_weight = 0.7  # 时间权重
    time_regular_weight = 0.5  # 融合权重
    interest_weight = 0.8  # 时间权重
    df_train = None
    df_test = None
    quiet = False
    test_df = None

    def __init__(self, limit=1000, source_type='sql', file_name='../待分析数据/for_analysis.csv', encoding='GBK',
                 recommend_count=30,
                 train_rate=1.0, test_rate=0, column_name=('用户号', '节目名称'), quiet=False):
        self.limit = limit
        self.recommend_count = recommend_count
        self.train_rate = train_rate
        self.test_rate = test_rate
        self.read_data(source_type, file_name, encoding)
        self.column_name = column_name
        self.quiet = quiet

    # 读取数据
    def read_data(self, source_type='sql', file_name='../待分析数据/for_analysis.csv', encoding='GBK'):
        if source_type == 'sql':
            engine = create_engine('mysql+pymysql://root:9q8w7e1108@127.0.0.1:3306/tv_distinct?charset=utf8')
            sql = pd.read_sql('select * from for_analysis_time', engine)
            sql.head()
            data = sql
        else:
            data = pd.read_csv(file_name, encoding=encoding)
        self.print('读取{}数据'.format(source_type))

        # 取部分数据
        # self.data = data[:1000]
        self.data = data

        # 取所需列
        clean_data = self.data[['用户号', '节目名称', '观看次数', '观看时长', '分类', '二级目录', '分类名称', '时间跨度', '使用时长', '总观看时长']]
        self.cleaned = clean_data
        if self.limit:
            self.cleaned = clean_data[:self.limit]

    # 生成训练与测试用数据
    def generate_train_data(self, train_rate=1.0, test_rate=0.0):
        self.print('生成训练与测试数据……')
        df = self.cleaned
        # 将数据随机打乱之后分成训练集和测试集

        simpler = np.random.permutation(len(df))
        df = df.take(simpler)  # 打乱数据

        # 取train_rate * 100%的数据作为训练数据
        train_range = int(len(df) * train_rate)
        train = df.iloc[:train_range].reset_index(drop=True)
        # train = df
        # 取test_rate * 100%之后的数据作为测试数据
        test_range = int(len(df) * test_rate)
        test = df.iloc[-test_range:].reset_index(drop=True)

        self.df_train = train
        self.df_test = test

        train.to_csv(self.output_dir + '1.训练数据.csv')
        test.to_csv(self.output_dir + '1.测试数据.csv')

    # 生成评分矩阵
    def generate_score_matrix(self):
        # 将数据转换成指标矩阵
        self.print('转换指标矩阵……')
        pre_data = self.df_train.copy()

        start = time.clock()

        pre_data.sort_values(by=[self.column_name[0], self.column_name[1]], ascending=[True, True], inplace=True)
        user_id = pre_data[self.column_name[0]].value_counts().index
        user_id = np.sort(user_id)
        programme = pre_data[self.column_name[1]].value_counts().index  #
        programme = np.sort(programme)
        data = pd.DataFrame([], index=user_id, columns=programme)
        for i in range(len(pre_data)):
            p = pre_data.loc[i, [self.column_name[0], self.column_name[1], '时间跨度', '使用时长', '观看时长', '总观看时长']]
            p1 = p[self.column_name[0]]
            p2 = p[self.column_name[1]]
            # 融合评分
            # 通过融合权重进行融合
            data.loc[p1, [p2]] = 1

        data = data.fillna(0)
        end = time.clock()

        # 由于基于物品的推荐，对于矩阵，根据上面的推荐函数，index应该为电视产品(即计算电视产品间的相似度），因此需要进行转置
        self.score_matrix = pd.DataFrame(data).T
        self.score_matrix.to_csv(self.output_dir + '2.1-0指标矩阵.csv')
        self.print('转换指标矩阵所用时间为：{}'.format(end - start))
        return self.score_matrix

    # 生成相似度矩阵
    def generate_similarity_matrix(self):
        self.print('生成相似度矩阵……')
        df_train = self.score_matrix.as_matrix()

        # 建立相似矩阵，训练模型
        start1 = time.clock()

        self.recommender.fit(df_train)  # 计算物品的相似度矩阵
        sim = self.recommender.sim
        end1 = time.clock()

        a = pd.DataFrame(sim)  # 保存相似度矩阵

        usetime1 = end1 - start1
        a.to_csv(self.output_dir + '3.相似度矩阵.csv')
        self.print(u'建立相似矩阵耗时' + str(usetime1) + 's')  # 5.345645981682992s

    # 生成推荐矩阵
    def predict(self):
        self.print('生成推荐矩阵……')
        # 使用测试集进行预测
        # test = pd.read_csv(self.output_dir + '1.测试数据.csv', index_col=0)
        # df_test = self.df_test.as_matrix()
        # 使用训练集预测，测试数据用于检查召回率与精确度
        # df_train = self.df_train.as_matrix()
        df_train = self.score_matrix.as_matrix()
        start2 = time.clock()
        result = self.recommender.recommend(df_train)
        end2 = time.clock()
        result1 = pd.DataFrame(result)
        usetime2 = end2 - start2
        # 将推荐结果表格中的对应的电视产品和用户名对应上
        result1.to_csv(self.output_dir + '4.推荐结果矩阵.csv')
        self.print(u'生成推荐矩阵耗时' + str(usetime2) + 's')

    def filter_recommend(self, recommend_count=20):
        self.print('使用协同过滤生成初步推荐结果……')
        result1 = pd.read_csv(self.output_dir + '4.推荐结果矩阵.csv', index_col=0)
        # 加载测试集
        # test = pd.read_csv(self.output_dir + '1.测试数据.csv', index_col=0)
        test = pd.read_csv(self.output_dir + '1.训练数据.csv', index_col=0)

        # 设置行索引与列名
        result1.index = test[self.column_name[1]].value_counts().index  # 节目名称
        result1.columns = test[self.column_name[0]].value_counts().index  # 用户号

        # 定义展现具体协同推荐结果的函数，K为推荐的个数，recomMatrix为协同过滤算法算出的推荐矩阵的表格化

        def xietong_result(K, recomMatrix):
            recomMatrix.fillna(0.0, inplace=True)  # 将表格中的空值用0填充
            recommends = ['结果', '推荐指数']  # 推荐结果列名
            currentemp = pd.DataFrame([], index=recomMatrix.columns, columns=recommends)

            row = 0
            for i in range(len(recomMatrix.columns)):
                temp = recomMatrix.sort_values(by=[recomMatrix.columns[i]], ascending=False)  # 降序排列，从最上面开始取电视产品就行了。
                k = 0
                while k < K:
                    if temp.iloc[k, i] < 0.5:  # 没有点击过相关电视产品
                        break
                    currentemp.iloc[row, 0] = str(temp.index[k])
                    currentemp.iloc[row, 1] = str(temp.iloc[k, i])
                    k = k + 1
                    row += 1

            return currentemp

        start3 = time.clock()

        xietong_result = xietong_result(recommend_count, result1)
        end3 = time.clock()
        self.print('使用协同过滤推荐方法为用户推荐' + str(recommend_count) + '个未浏览过的电视产品，耗时为' + str(end3 - start3) + 's')

        xietong_result = xietong_result.reset_index()
        xietong_result.columns = [self.column_name[0], '结果', '推荐指数']

        xietong_result.to_csv(self.output_dir + '5.协同过滤结果.csv')

    def final(self):
        self.print('生成最终程序……')

        result = pd.read_csv(self.output_dir + '5.协同过滤结果.csv')

        # dataset = self.data
        # result['观看记录'] = ''
        # clean_data = dataset[['用户号', '节目名称', '分类', '分类名称', '二级目录', '观看次数', '观看时长', '时间跨度', '使用时长']]
        # result = result[[self.column_name[0], '观看记录', '结果', '推荐指数']]
        # user_id = result[[self.column_name[0]]]
        # for index in user_id.index:
        #     yhh = str(result.iloc[index, 0])
        #     user_data = clean_data[clean_data[self.column_name[0]].str.match(yhh)]
        #     result.iloc[index, 1] = user_data[self.column_name[1]].str.cat(sep='\n')

        final_list = []
        # k = 0
        for index in result.index:
            if str(result.iloc[index, 2]) == 'nan' or str(result.iloc[index, 3]) == 'nan':
                continue
            movies = str(result.iloc[index, 2]).split('\n')
            scores = str(result.iloc[index, 3]).split('\n')
            for i in range(len(movies)):
                # final = final.append({self.column_name[0]: result.iloc[index, 1],
                # '产品名称': movies[i], '推荐指数': scores[i]})
                if movies[i] and scores[i]:
                    final_list.append(
                        {self.column_name[0]: result.iloc[index, 1], '产品名称': movies[i], '推荐指数': scores[i]})
                # k += 1

        final = pd.DataFrame(final_list)[[self.column_name[0], '产品名称', '推荐指数']]
        result = final
        result.to_csv(self.output_dir + '6.最终推荐表_GBK.csv', index=False, encoding='GBK')
        result.to_csv(self.output_dir + '6.最终推荐表.csv', index=False)
        result.to_excel(self.output_dir + '6.最终推荐表.xlsx', index=False)
        self.print('推荐结果生成至：' + self.output_dir + '6.最终推荐表.csv')

    # 召回率与准确率
    def recall_precision(self):
        test_data = pd.read_csv(self.output_dir + '1.测试数据.csv', index_col=0)
        test_data.sort_values(by=[self.column_name[0], self.column_name[1]], ascending=[True, True], inplace=True)
        result = pd.read_csv(self.output_dir + '6.最终推荐表.csv', dtype=np.str)
        hit = 0
        recall = 0
        precision = 0
        user_ids = test_data[self.column_name[0]].value_counts().index
        programmes = test_data[self.column_name[1]].value_counts().index
        for user_id in user_ids:
            # 循环用户，得到用户看过的节目名称
            user_test = test_data[test_data[self.column_name[0]] == user_id]
            user_test_movies = user_test[self.column_name[1]].value_counts().index
            try:
                user_result = result[result[self.column_name[0]] == str(user_id)]['产品名称'].value_counts().index
                for movie in user_result:
                    if movie in user_test_movies:
                        hit += 1
            except KeyError:
                user_result = None

            recall += len(user_test)
            precision += self.recommend_count
        recall = hit / (recall * 1.0)
        precision = hit / (precision * 1.0)
        return recall, precision

    # 推荐覆盖率
    def coverage(self):
        train_data = pd.read_csv(self.output_dir + '1.训练数据.csv', index_col=0)
        result = pd.read_csv(self.output_dir + '6.最终推荐表.csv', dtype=np.str)

        programmes = train_data[self.column_name[1]].value_counts().index
        # all_items = set()  # 所有物品
        all_items = programmes  # 所有物品

        # for movie in programmes:
        #     all_items.add(movie)

        # user_result = result[['产品名称']].values.reshape(-1)
        # for movie in user_result:
        #     recommended_items.add(movie)
        recommended_items = result['产品名称'].value_counts().index

        return 1.0 * len(recommended_items) / len(all_items)

    def generate(self):
        if self.data is None:
            raise ValueError('还未读取数据')
        self.generate_train_data(train_rate=self.train_rate, test_rate=self.test_rate)
        self.generate_score_matrix()
        self.generate_similarity_matrix()
        self.predict()

    def procedure(self, generated=True):
        start_time = time.clock()
        if not generated:
            self.generate()
        self.filter_recommend(recommend_count=self.recommend_count)
        self.final()
        end_time = time.clock()
        self.print('程序执行完毕，耗时{}s'.format(end_time - start_time))

    def test(self):
        recall, precision = self.recall_precision()
        coverage = self.coverage()
        result = [{
            'K': self.recommend_count,
            '召回率': recall,
            '准确率': precision,
            '推荐覆盖率': coverage
        }]
        if self.test_df is None:
            df = pd.DataFrame({}, columns=['K', '召回率', '准确率', '推荐覆盖率'])
        else:
            df = self.test_df

        df = df.append(result)

        self.test_df = df

        df.to_csv(self.output_dir + '7.模型准确率.csv', index=False)
        df.to_csv(self.output_dir + '7.模型准确率_GBK.csv', encoding='GBK', index=False)
        return df.set_index(['K'])

    def print(self, obj):
        if self.quiet is True:
            return False
            pass
        else:
            return print(obj)
