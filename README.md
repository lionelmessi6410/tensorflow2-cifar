# Training CIFAR-10 with TensorFlow2(TF2)
[![TensorFlow 2.4](https://img.shields.io/badge/TensorFlow-2.4-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.4.0)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB)](https://www.python.org/downloads/release/python-380/)
[![License](https://img.shields.io/github/license/lionelmessi6410/tensorflow2-cifar)](https://github.com/lionelmessi6410/tensorflow2-cifar/blob/main/LICENSE)

TensorFlow2 Classification Model Zoo. I'm playing with [TensorFlow2](https://www.tensorflow.org/) on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Architectures
- [LeNet](https://ieeexplore.ieee.org/abstract/document/726791)
- [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [VGG](https://arxiv.org/abs/1409.1556) [11, 13, 16, 19]
- [ResNet](https://arxiv.org/abs/1512.03385) [18, 34, 50, 101, 152]
- [DenseNet](https://arxiv.org/abs/1608.06993) [121, 169, 201]
- [PreAct-ResNet](https://arxiv.org/abs/1603.05027) [18, 34, 50, 101, 152]
- [SENet](https://arxiv.org/abs/1709.01507)
- [SE-ResNet](https://arxiv.org/abs/1709.01507) [18, 34, 50, 101, 152]
- [SE-PreAct-ResNet](https://arxiv.org/abs/1709.01507) [18, 34, 50, 101, 152]
- [MobileNet](https://arxiv.org/abs/1704.04861) 
- [MobileNetV2](https://arxiv.org/abs/1801.04381)

## Prerequisites
- Python 3.8+
- TensorFlow 2.4.0+

## Training
Start training with:
```python
python train.py --model resnet18
```

You can manually resume the training with: 
```python
python train.py --model resnet18 --resume
```

## Testing 
```python
python test.py --model resnet18
```

## Accuracy
| Model               | Acc.   | Param. |
| ------------------- | ------ | -----: |
| LeNet               | 67.85% |  0.06M |
| AlexNet             | 78.81% |  21.6M |
| VGG11               | 92.61% |   9.2M |
| VGG13               | 94.31% |   9.4M |
| VGG16               | 94.27% |  14.7M |
| VGG19               | 93.65% |  20.1M |
| ResNet18            | 95.37% |  11.2M |
| ResNet34            | 95.48% |  21.3M |
| ResNet50            | 95.41% |  23.6M |
| ResNet101           | 95.44% |  42.6M |
| ResNet152           | 95.29% |  58.3M |
| DenseNet121         | 95.37% |   7.0M |
| DenseNet169         | 95.10% |  12.7M |
| DenseNet201         | 94.79% |  18.3M |
| PreAct-ResNet18     | 94.08% |  11.2M |
| PreAct-ResNet34     | 94.76% |  21.3M |
| PreAct-ResNet50     | 94.81% |  23.6M |
| PreAct-ResNet101    | 94.95% |  42.6M |
| PreAct-ResNet152    | 95.07% |  58.3M |
| SE-ResNet18         | 95.44% |  11.3M |
| SE-ResNet34         | 95.30% |  21.5M |
| SE-ResNet50         | 95.76% |  26.1M |
| SE-ResNet101        | 95.40% |  47.3M |
| SE-ResNet152        | 95.29% |  64.9M |
| SE-PreAct-ResNet18  | 94.54% |  11.3M |
| SE-PreAct-ResNet34  | 95.30% |  21.5M |
| SE-PreAct-ResNet50  | 94.22% |  26.1M |
| SE-PreAct-ResNet101 | 94.34% |  47.3M |
| SE-PreAct-ResNet152 | 94.28% |  64.9M |
| MobileNet           | 92.34% |   3.2M |
| MobileNetV2         | 94.03% |   2.3M |


## Note
All abovementioned models are available. To specify the model, please use the model name without the hyphen. For instance, to train with `SE-PreAct-ResNet18`, you can run the following script:

```python
python train.py --model sepreactresnet18
```

If you suffer from `loss=nan` issue, you can circumvent it by using a smaller learning rate, i.e.
```python
python train.py --model sepreactresnet18 --lr 5e-2
```