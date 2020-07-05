## DSFaceNet

###  运行环境

- Python 3.x
- Pytorch 1.x
- NVIDIA GPU 2080Ti

### 运行方式

#### 1、下载数据集

[Align-CASIA-WebFace@BaiduDrive](https://pan.baidu.com/s/1k3Cel2wSHQxHO9NkNi3rkg) and [Align-LFW@BaiduDrive.](https://pan.baidu.com/s/1r6BQxzlFza8FM8Z8C_OCBg)

这个数据集中是已经对齐的人脸，图片尺寸为 H:112 W:96

#### 2、训练

在`config.py`文件中设置好配置项

输入下面的命令训练

```bash
python train.py
```

#### 3、测试

在LFW数据集上进行测试

```shell script
sh ./run.sh
```

#### 4、运行结果


```python
平均每张照片的推理速度:8 ms
1    98.50
2    99.17
3    98.50
4    98.67
5    98.17
6    99.17
7    98.67
8    99.00
9    99.83
10    99.67
--------
AVE    98.93
```

在对齐的LFW数据集十折交叉验证的平均精准度为98.93%，平均推断一组图片所用时间为8ms


