
## 介绍
NLPworker是一个深度学习快速开发平台，它将大多数项目中共有的代码写入底层框架，
用户只需要针对自己的项目修改少量的代码即可。对于不同的模型或任务类型提供了测试样例，
对于相似的项目可以进行更少的修改。

对于常用的机器学习模型、评价指标、分词器等等进行复现并详细注释，以便初学者可以更好的学习，
项目也在不断完善中。

## 项目结构

- examples
    - BiLSTM+CRF：使用BiLSTM+CRF完成ner任务

- theshy：TheShy是我比较喜欢的一个LPL选手，框架就以此来命名
  - base：模型训练、验证、推理的base文件，主要是项目共有的代码
    - config：配置base文件，设置配置默认值，保证examples未设置时不报错
    - train：初始化优化器等信息，完成训练通用流程编写，自动保存指定checkpoint
    - evaluate：通用验证流程
  - metrics：复现的评价指标
    - ner：ner任务评价指标，包括token级别和实体级别评价指标
  - model：复现的机器学习模型
    - crf
    - pageRank
    - tfidf
  - utils：工具文件
    - _tokenize
      - basicTokenizer：基础分词器
      - BPE：BPE分词器