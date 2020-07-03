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




