import TVRecommend
import time

start_time = time.clock()
print('改进模型')
print('开始于：{}'.format(start_time))

r = TVRecommend.TVRecommend(file_name='for_analysis_time.csv', encoding='GBK',
                            limit=False, recommend_count=20,
                            source_type='file', column_name=('用户号', '地区'),
                            train_rate=1, test_rate=0,
                            quiet=True)


# r = TVRecommend.TVRecommend(file_name='for_analysis_time.csv', encoding='GBK',
#                             limit=False, recommend_count=20,
#                             source_type='file', column_name=('用户号', '节目名称'),
#                             train_rate=0.7, test_rate=0.3,
#                             quiet=True)


# r = TVRecommend.TVRecommend(limit=100, recommend_count=5,
#                             source_type='sql', column_name=('用户号', '节目名称'),
#                             train_rate=0.7, test_rate=0.3,
#                             quiet=True)


def test_k():
    # r.recommend_count = 5
    # r.procedure(generated=False)
    # r.test()
    r.recommend_count = 10
    r.procedure(generated=False)
    print('运行中，K：{}，时间权重增长系数：{}，兴趣权重增长系数：{}，融合比例：{}'.format(r.recommend_count, r.time_weight, r.interest_weight,
                                                            r.time_regular_weight))
    # result = r.test()
    # r.recommend_count = 20
    # r.procedure(generated=True)
    # r.test()
    # r.recommend_count = 30
    # r.procedure(generated=True)
    # r.test()
    # r.recommend_count = 40
    # r.procedure(generated=True)
    result = r.test()
    return result


def test_time_weight():
    # r.time_weight = 0.3
    # test_k()
    # r.time_weight = 0.4
    # test_k()
    # r.time_weight = 0.5
    # test_k()
    # r.time_weight = 0.6
    # test_k()
    r.time_weight = 0.7
    return test_k()


def test_time():
    # r.interest_weight = 0.3
    # test_time_weight()
    # r.interest_weight = 0.4
    # test_time_weight()
    # r.interest_weight = 0.5
    # test_time_weight()
    # r.interest_weight = 0.6
    # test_time_weight()
    r.interest_weight = 0.7
    return test_time_weight()


# 融合比例
# r.time_regular_weight = 0.3
# result = test_time()
# r.time_regular_weight = 0.4
# result = test_time()
# r.time_regular_weight = 0.5
# result = test_time()
# r.time_regular_weight = 0.6
# result = test_time()
# r.time_regular_weight = 0.7
# result = test_time()
r.time_regular_weight = 0.8
result = test_time()
print(result)
# r.quiet = False
# r.procedure(generated=True)
# r.procedure(generated=False)
end_time = time.clock()
duration = end_time - start_time
print('结束于：{}，耗时：{}s'.format(end_time, duration))
