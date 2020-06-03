import TVRecommend
import time

start_time = time.clock()
print('传统协同过滤')
print('开始于：{}'.format(start_time))

r = TVRecommend.TVRecommend(file_name='for_analysis_time.csv', encoding='GBK',
                            limit=False, recommend_count=5,
                            source_type='file', column_name=('用户号', '节目名称'),
                            train_rate=0.7, test_rate=0.3, quiet=True)


# r = TVRecommend.TVRecommend(limit=100, recommend_count=5,
#                             source_type='sql', column_name=('用户号', '节目名称'),
#                             train_rate=0.7, test_rate=0.3)


def test_k():
    print('运行中，K：{}'.format(r.recommend_count))
    r.recommend_count = 5
    r.procedure(generated=True)
    r.test()
    r.recommend_count = 10
    r.procedure(generated=True)
    r.test()
    r.recommend_count = 20
    r.procedure(generated=True)
    r.test()
    r.recommend_count = 40
    r.procedure(generated=True)
    return r.test()


print(test_k())

# r.test(head=False)
# r.procedure(generated=True)
# print(r.generate_score_matrix())
# print(r.generate_interest_matrix())

end_time = time.clock()
duration = end_time - start_time
print('结束于：{}，耗时：{}s'.format(end_time, duration))
