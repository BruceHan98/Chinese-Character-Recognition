# Chinese-Character-Recognition
中文手写字符识别，CASIA-HWDB1.x 数据集，pytorch实现。

#### 1. 数据准备

数据集为中科院CASIA-HWDB1.0-1.2 (.gnt) 单字符数据集，更多说明参见[数据集](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html)。

1. 下载相应数据集并解压到指定文件夹下（配置参见 config.py）。
2.  运行 utils/gnt_parser.py ，解析 .gnt 文件并生成训练数据。

#### 2. 环境配置

​	pip install -r requirements.txt

#### 3. 运行

​	python main.py --use_gpu

可根据网络结构和数据调整学习策略。
