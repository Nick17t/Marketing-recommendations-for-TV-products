import numpy as np


# 杰卡德相似系数
def Jaccard(a, b):  # 自定义杰卡德相似系数函数，仅对0-1矩阵有效
    if (a + b - a * b).sum() == 0:
        return 0
    return 1.0 * (a * b).sum() / (a + b - a * b).sum()


# 欧氏距离
def eulid_sim(col_a, col_b):
    return 1.0 / (1.0 + np.linalg.norm(col_a - col_b))


# 皮尔逊相关系数
def pearson_sim(col_a, col_b):
    if len(col_a) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(col_a, col_b, rowvar=0)[0][1]


class Recommender:
    sim = None  # 相似度矩阵

    def similarity(self, x, distance):  # 计算相似度矩阵的函数
        y = np.ones((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                y[i, j] = distance(x[i], x[j])
        return y

    def fit(self, x, distance=Jaccard):  # 训练函数
        self.sim = self.similarity(x, distance)

    def recommend(self, a):  # 推荐函数
        return np.dot(self.sim, a) * (1 - a)
