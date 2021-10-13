# Training CIFAR-10 with TensorFlow2(TF2)
I'm playing with [TensorFlow](https://www.tensorflow.org/) on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Prerequisites
- Python 3.8+
- TensorFlow 2.4.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume
```

## Accuracy
| Model             | Acc.        | Param.        |
| ----------------- | ----------- | ------------: |
| [VGG11](https://arxiv.org/abs/1409.1556)              | 92.61% | 9.2M |
| [VGG13](https://arxiv.org/abs/1409.1556)              | 94.31% | 9.4M |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 94.27% | 14.7M |
| [VGG19](https://arxiv.org/abs/1409.1556)              | 93.65% | 20.1M |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 95.37% | 11.2M |
| [ResNet34](https://arxiv.org/abs/1512.03385)          | 95.48% | 21.3M |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 95.41% | 23.6M |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 95.44% | 42.6M |
| [ResNet152](https://arxiv.org/abs/1512.03385)         | 95.29% | 58.3M |
<!-- | [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      | -->
