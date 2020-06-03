import SuitRecommend
import time

start_time = time.clock()
print('传统协同过滤，用户套餐')
print('开始于：{}'.format(start_time))

r = SuitRecommend.TVRecommend(file_name='suit_for_analysis.csv', encoding='GBK',
                              limit=False, recommend_count=5,
                              source_type='file',
                              train_rate=1.0, test_rate=1.0,
                              quiet=True)

r.recommend_count = 1
print('运行中，K：{}'.format(r.recommend_count))
r.procedure(generated=False)

end_time = time.clock()
duration = end_time - start_time
print('结束于：{}，耗时{}'.format(end_time, duration))
