# PSPNet-tensorflow-chapter1

The chapter1 of the segmentation network summary: The traditional typical segmentation network.

Github has many different versions of segmentation network like Deeplab,SegNet, etc. Here, I only published the code of PSPNet modified and debugged by myself, and used it for the training and test of my own data set (since the data set belongs to the laboratory and cannot be leaked, I did not add the test results of data).

External links: Pyramid Scene Parsing Network [paper](https://arxiv.org/abs/1612.01105) and [official github](https://github.com/hszhao/PSPNet).

Here I would like to thank [holyseven](https://github.com/holyseven) for using PSPNet on my dataset by modifying his code. Please read the [details](https://github.com/holyseven/PSPNet-TF-Reproduce).

# Notes

1. In addition to the new ASPP structure proposed by the network, there is also an auxiliary loss in the network. It is verified by our own experiments that the auxiliary loss can indeed accelerate the model convergence.

2. In fact, the network can be divided into two parts, one is the resnet and the other is the structure proposed by the author. On my own dataset, I first use resnet to train my data to get the pre-training model, and then use the model proposed by the author to fine-tune the pre-training model. The mIoU obtained by such processing is higher than that obtained by direct training. And this training tip can also be used on other networks.
