# TextClassification
文本分类中常用的神经网络模型
- 卷积神经网络
- 循环神经网络(RNN,LSTM,GRU,SRU)
- 自定义模型

## 环境

- Python 3
- pycharm 
- TensorFlow 1.7
- pandas
- numpy
- scikit-learn

## 数据集

采用清华大学中文文本分类数据集[THUCNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews)
THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。数据集被划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

  
|Data|Shape|Sequence|Language|
| ------ | ------ | :------: | :------: |
| 训练集 | (50000,) |600| CN |
| 验证集 | (5000,) |600| CN |
| 测试集 | (10000,) |600| CN |
  
