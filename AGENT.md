Answer me in Chinese!
I am completing the Image Processing Course, Read the Project Description First! (I choose the fifth topic)

第五个课题概述（Mapping Swiss Ecosystems from Aerial Images and Environmental Variables）：
- 目标：结合航拍 RGB 影像和 SWECO25 的 48 个环境变量，预测 17 类 EUNIS 生态系统。
- 数据：16925 个样本，每个样本含 100×100 m、50 cm 分辨率的 swissIMAGE RGB 图 + 48 个已标准化数值变量；地理划分的 train 60% / val 10% / test 30%。
- 期望产出：搭建训练管线（预训练 backbone、损失、调参），在测试集报告性能；做按主题的变量子集消融（影像 vs 环境 vs 融合），分析贡献；用图/地图展示与解释结果。
- 关键挑战：地理分布差异、影像与表格数据的融合方式、17 类不平衡、多模态贡献解释。
