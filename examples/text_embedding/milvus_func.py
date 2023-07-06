#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/6/23 17:08
# @File     : milvus_func.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : milvus库拓展文件


# 需要先安装docker，拉取docker官网上的公共镜像milvus，docker安装参考：https://blog.csdn.net/weixin_50999155/article/details/119581698
# 配置milvus教程：https://zhuanlan.zhihu.com/p/166454687
# milvus官网操作手册：https://milvus.io/docs


# pymilvus不同版本（1.x和2.x）函数差异较大，引入方式也不同，这里使用1.1.0版本
from milvus import Milvus, MetricType

# 端口由docker启动时指定映射
milvus = Milvus(host='localhost', port='19530')


class MyMilvus:
    def __init__(self, collection_name, vec_dim, drop=True):
        self.collection_name = collection_name
        collection_param = {
            'collection_name': collection_name,
            'dimension': vec_dim,
            'index_file_size': 32,  # 数据导入时的缓冲区大小
            'metric_type': MetricType.IP,  # 使用内积作为度量值
            'auto_id': True
        }
        # 如果该集合已存在，则删除集合后再重新创建
        # 或使用 milvus.has_collection(collection_name)[1]，存在返回True，不存在返回False
        if milvus.get_collection_stats(collection_name)[0].OK():
            if drop:
                milvus.drop_collection(collection_name)
                print(f'{collection_name}已存在，现已删除并重新创建')
            else:
                raise Exception(f'{collection_name}已存在，请更改集合名或设置drop=True')

        status = milvus.create_collection(collection_param)
        if status.OK():
            print(f'成功创建{collection_name}')
        else:
            raise Exception(f'创建{collection_name}失败')

        # 用于存储id2index的json文件
        self.id2index = {}

    def insert_vectors(self, vector_list):
        # milvus搜索采用倒排索引模型
        status, ids = milvus.insert(collection_name=self.collection_name, records=vector_list)
        self.id2index.update({id: index for index, id in enumerate(ids)})
        # 从内存中刷新集合中的数据以使数据可用，Milvus每隔1秒钟自动刷新所有现有数据
        milvus.flush([self.collection_name])

if __name__ == '__main__':
    # 删除某个向量
    milvus.delete_entity_by_id(collection_name='glove_300d', id_array=[1687601053399834808])
    # 关闭客户端
    milvus.close()