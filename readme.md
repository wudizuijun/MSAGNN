### 本项目亦在设计一个通用的深度学习预测框架
预期效果， 给定一个mode_list, 用户选择其中的model, 之后使用这些模型进行预测，输出对应的结果
- 有共性的model 单独放在一块， 没有的单独调用

#### output format:
file_path: results:
datetime/folder_name_r2_mse/[res_mode_i.csv for i in len(models)]
res_mode_i.csv formate: columns -> [pred, real]

#### 不同模型进行预测主要进行修改内容:
1. 确定数据集， 包括 X, y, timestamp(数据一般为时间序列)
2. 明确模型，本框架适用seq2seq的模型


#### 操作：
主要包括2大部分，第一步为模型训练以及预测，第二步为绘图

##### 绘图：
判断数据位置： datetime
遍历所有文件夹， 把所有能绘制的图都给绘制了
file_path
- select multi_model or single model
  - 