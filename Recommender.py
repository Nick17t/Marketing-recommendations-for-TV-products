import numpy as np


# 杰卡德相似系数
def Jaccard(a, b):  # 自定义杰卡德相似系数函数，仅对0-1矩阵有效
    if (a + b - a * b).sum() == 0:
        return 0
    return 1.0 * (a * b).sum() / (a + b - a * b).sum()


# 欧氏距离
def eulid_sim(col_a, col_b):
    return 1.0 / (1.0 + np.linalg.norm(col_a - col_b))


# 夹角余弦
def cos_sim(a, b):
    num = (a * b.T).sum()  # 若为行向量则 a*b.T
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0
    cos = num / denom  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return sim


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

    def fit(self, x, distance=cos_sim):  # 训练函数
        self.sim = self.similarity(x, distance)
        return self.sim

    def recommend(self, a):  # 推荐函数
        return np.dot(self.sim, a) * (1 - a)
