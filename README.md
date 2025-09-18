# 📝 Assignment : MNIST Classification With less than 20K Parameters and Achieve 99.4% Accuracy In Less Than 20 Epochs.

## 🎯 Challenge

The goal is to train a neural network on the MNIST dataset that:

- Uses fewer than 20,000 trainable parameters
- Achieves ≥99.4% test accuracy
- Converges in fewer than 20 epochs

## 📊 Dataset

MNIST: 70,000 grayscale images of handwritten digits (0–9), each of size 28×28 pixels.
- Training set: 60,000 images
- Test set: 10,000 images

## 🔄 Data Augmentation
To improve generalization and prevent overfitting, we apply augmentations during training:
- Random Center Crop (p=0.1): randomly zooms in on digits
- Resize to (28×28): ensures input consistency
- Random Rotation (±10°): handles tilted handwriting
- Color Jitter (brightness, contrast): adds lighting variations
- Random Affine Transform: translation (±10%), scaling (0.9–1.1)

These transformations mimic real-world handwriting variations and make the model robust.

```
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale = (0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
```

## Model Architecture
```
 Net(
  (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (drop1): Dropout(p=0.05, inplace=False)
  (conv3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn3): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv4): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (drop2): Dropout(p=0.05, inplace=False)
  (conv5): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv6): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn6): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (drop3): Dropout(p=0.05, inplace=False)
  (gap): AdaptiveAvgPool2d(output_size=1)
  (fc): Linear(in_features=32, out_features=10, bias=True)
)
```
The model is a small CNN with careful design to stay under 20K parameters.
- Conv1 (1→8 channels, 3×3, BN, ReLU)
- Conv2 (8→16 channels, 3×3, BN, ReLU) + MaxPool + Dropout
- Conv3 (16→16 channels, 3×3, BN, ReLU)
- Conv4 (16→32 channels, 3×3, BN, ReLU) + MaxPool + Dropout
- Conv5 (32→32 channels, 3×3, BN, ReLU)
- Conv6 (32→32 channels, 1×1, BN, ReLU) + Dropout
- Global Average Pooling (GAP) → reduces each feature map to a single value
- Fully Connected Layer (32→10)

### 🔑 Key design choices:
- Batch Normalization → stabilizes and accelerates training
- Dropout (0.05–0.1) → combats overfitting despite small size
- Global Average Pooling → drastically reduces parameters vs. dense layers
- 1×1 Convolution → reduces complexity without losing representation power

### 📉 Model Summary and Parameters 
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
            Conv2d-3           [-1, 16, 28, 28]           1,168
       BatchNorm2d-4           [-1, 16, 28, 28]              32
            Conv2d-5           [-1, 16, 14, 14]           2,320
       BatchNorm2d-6           [-1, 16, 14, 14]              32
            Conv2d-7           [-1, 32, 14, 14]           4,640
       BatchNorm2d-8           [-1, 32, 14, 14]              64
            Conv2d-9             [-1, 32, 7, 7]           9,248
      BatchNorm2d-10             [-1, 32, 7, 7]              64
           Conv2d-11             [-1, 32, 7, 7]           1,056
      BatchNorm2d-12             [-1, 32, 7, 7]              64
          Dropout-13             [-1, 32, 7, 7]               0
AdaptiveAvgPool2d-14             [-1, 32, 1, 1]               0
           Linear-15                   [-1, 10]             330
================================================================
Total params: 19,114
Trainable params: 19,114
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.49
Params size (MB): 0.07
Estimated Total Size (MB): 0.57
----------------------------------------------------------------
```
## ⚙️ Training Setup
- Optimizer: Adam (lr=0.001) → adaptive learning rates for fast convergence
- Scheduler: StepLR (step_size=15, gamma=0.5) → reduces learning rate when training plateaus
- Loss Function: CrossEntropyLoss (standard for multi-class classification)
- Batch Size: 128
- Epochs: 20

## 📈 Results
- Convergence: Achieved 99.4% test accuracy in fewer than 20 epochs
- Efficiency: <20K parameters, training completes quickly on GPU
- Generalization: Augmentations + regularization prevent overfitting
```
Epoch 1
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
Train: Loss=0.1463 Batch_id=468 Accuracy=83.06: 100%|██████████| 469/469 [00:35<00:00, 13.21it/s]
Test set: Average loss: 0.0009, Accuracy: 9768/10000 (97.68%)

Epoch 2
Train: Loss=0.2077 Batch_id=468 Accuracy=97.23: 100%|██████████| 469/469 [00:34<00:00, 13.54it/s]
Test set: Average loss: 0.0005, Accuracy: 9832/10000 (98.32%)

Epoch 3
Train: Loss=0.0279 Batch_id=468 Accuracy=97.90: 100%|██████████| 469/469 [00:34<00:00, 13.63it/s]
Test set: Average loss: 0.0004, Accuracy: 9859/10000 (98.59%)

Epoch 4
Train: Loss=0.0318 Batch_id=468 Accuracy=98.03: 100%|██████████| 469/469 [00:34<00:00, 13.66it/s]
Test set: Average loss: 0.0003, Accuracy: 9908/10000 (99.08%)

Epoch 5
Train: Loss=0.1191 Batch_id=468 Accuracy=98.34: 100%|██████████| 469/469 [00:34<00:00, 13.47it/s]
Test set: Average loss: 0.0002, Accuracy: 9908/10000 (99.08%)

Epoch 6
Train: Loss=0.0160 Batch_id=468 Accuracy=98.44: 100%|██████████| 469/469 [00:35<00:00, 13.39it/s]
Test set: Average loss: 0.0003, Accuracy: 9889/10000 (98.89%)

Epoch 7
Train: Loss=0.0467 Batch_id=468 Accuracy=98.56: 100%|██████████| 469/469 [00:35<00:00, 13.27it/s]
Test set: Average loss: 0.0002, Accuracy: 9919/10000 (99.19%)

Epoch 8
Train: Loss=0.0212 Batch_id=468 Accuracy=98.65: 100%|██████████| 469/469 [00:34<00:00, 13.42it/s]
Test set: Average loss: 0.0003, Accuracy: 9873/10000 (98.73%)

Epoch 9
Train: Loss=0.0381 Batch_id=468 Accuracy=98.75: 100%|██████████| 469/469 [00:35<00:00, 13.30it/s]
Test set: Average loss: 0.0002, Accuracy: 9922/10000 (99.22%)

Epoch 10
Train: Loss=0.0499 Batch_id=468 Accuracy=98.69: 100%|██████████| 469/469 [00:34<00:00, 13.52it/s]
Test set: Average loss: 0.0002, Accuracy: 9908/10000 (99.08%)

Epoch 11
Train: Loss=0.0316 Batch_id=468 Accuracy=98.79: 100%|██████████| 469/469 [00:33<00:00, 13.84it/s]
Test set: Average loss: 0.0002, Accuracy: 9917/10000 (99.17%)

Epoch 12
Train: Loss=0.0461 Batch_id=468 Accuracy=98.80: 100%|██████████| 469/469 [00:33<00:00, 13.87it/s]
Test set: Average loss: 0.0002, Accuracy: 9926/10000 (99.26%)

Epoch 13
Train: Loss=0.0650 Batch_id=468 Accuracy=98.92: 100%|██████████| 469/469 [00:34<00:00, 13.47it/s]
Test set: Average loss: 0.0002, Accuracy: 9929/10000 (99.29%)

Epoch 14
Train: Loss=0.0719 Batch_id=468 Accuracy=98.99: 100%|██████████| 469/469 [00:34<00:00, 13.46it/s]
Test set: Average loss: 0.0002, Accuracy: 9918/10000 (99.18%)

Epoch 15
Train: Loss=0.0351 Batch_id=468 Accuracy=98.94: 100%|██████████| 469/469 [00:34<00:00, 13.42it/s]
Test set: Average loss: 0.0001, Accuracy: 9942/10000 (99.42%)

Epoch 16
Train: Loss=0.0115 Batch_id=468 Accuracy=99.17: 100%|██████████| 469/469 [00:34<00:00, 13.45it/s]
Test set: Average loss: 0.0001, Accuracy: 9940/10000 (99.40%)

Epoch 17
Train: Loss=0.0635 Batch_id=468 Accuracy=99.14: 100%|██████████| 469/469 [00:34<00:00, 13.45it/s]
Test set: Average loss: 0.0001, Accuracy: 9941/10000 (99.41%)

Epoch 18
Train: Loss=0.0085 Batch_id=468 Accuracy=99.22: 100%|██████████| 469/469 [00:34<00:00, 13.54it/s]
Test set: Average loss: 0.0001, Accuracy: 9950/10000 (99.50%)

Epoch 19
Train: Loss=0.0346 Batch_id=468 Accuracy=99.13: 100%|██████████| 469/469 [00:33<00:00, 13.91it/s]
Test set: Average loss: 0.0001, Accuracy: 9943/10000 (99.43%)

Epoch 20
Train: Loss=0.0033 Batch_id=468 Accuracy=99.21: 100%|██████████| 469/469 [00:34<00:00, 13.69it/s]
Test set: Average loss: 0.0001, Accuracy: 9949/10000 (99.49%)
```
