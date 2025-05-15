import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import gc

import paddle as paddle
import paddle.nn.functional as F
from paddle.vision.models.vgg import make_layers

from conv_next import ConvNeXt
from swin_transformer import SwinTransformer
from dataset_station import StationType

paddle.set_device('gpu:0')

# datasets
train_dataset = StationType(mode='train')
val_dataset = StationType(mode='val')

train_loader = paddle.io.DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=False)
val_loader = paddle.io.DataLoader(val_dataset, batch_size=4, shuffle=True, drop_last=False)
trainLoss, valLoss = [], []
trainAcc, valAcc = [], []

# model=paddle.Model(ConvNeXt(num_classes=10, num_patch = 8))
# model.summary((-1,3,1024,1024))
# model = CMT(num_classes=10)


# model=ConvNeXt(num_classes=10, num_patch=0)
# model=SwinTransformer(num_classes=10)
# DenseNet121 = paddle.vision.models.DenseNet(layers=121,num_classes=10)
# MobileNetV3 = paddle.vision.models.MobileNetV3Large(scale=1.0, num_classes=3)
# ShuffleNetV2 = paddle.vision.models.ShuffleNetV2(scale=1.0, act='relu', num_classes=3)
# SqueezeNet = paddle.vision.models.SqueezeNet(version = '1.0', num_classes=10)

vgg19_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
features = make_layers(vgg19_cfg)
VGG = paddle.vision.models.VGG(features, num_classes=10)

# ResNet50 = paddle.vision.models.resnet50(num_classes=10, pretrained=False)
# model=ResNet50

model = VGG
optimizer = paddle.optimizer.Adam(learning_rate=0.000001, parameters=model.parameters())
loss_fn = paddle.nn.loss.CrossEntropyLoss()

early_stop_patience = 5
early_stop_delta = 0.01
best_val_loss = np.inf
early_stop_counter = 0

import time

start_time = time.time()

for epoch in range(100):

    listTrainLoss = []
    listacc = []
    for batch, (train_X, train_y) in enumerate(train_loader):
        y_pred = model(train_X)
        loss = loss_fn(y_pred, train_y)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        listTrainLoss.append(float(loss))
        acc = paddle.metric.accuracy(y_pred, train_y)
        listacc.append(float(acc))

    arrayTrainLoss = np.array(listTrainLoss)
    arrayaccuracies = np.array(listacc)
    print('Train-epoch:{},loss:{}'.format(epoch, arrayTrainLoss.mean()))
    print('Train-epoch:{},acc:{}'.format(epoch, arrayaccuracies.mean()))
    trainLoss.append(arrayTrainLoss.mean())
    trainAcc.append(arrayaccuracies.mean())

    listValLoss = []
    listValAcc = []
    for batch, (val_X, val_y) in enumerate(val_loader):
        y_pred = model(val_X)
        loss = loss_fn(y_pred, val_y)
        listValLoss.append(float(loss))
        acc = paddle.metric.accuracy(y_pred, val_y)
        listValAcc.append(float(acc))

    arrayValLoss = np.array(listValLoss)
    arrayValAcc = np.array(listValAcc)
    current_val_loss = arrayValLoss.mean()
    print('Val-epoch:{},loss:{}'.format(epoch, arrayValLoss.mean()))
    print('Val-epoch:{},acc:{}'.format(epoch, arrayValAcc.mean()))
    valLoss.append(arrayValLoss.mean())
    valAcc.append(arrayValAcc.mean())

    # Early stopping check
    if current_val_loss < best_val_loss - early_stop_delta:
        print(f'Validation loss improved from {best_val_loss:.4f} to {current_val_loss:.4f}')
        best_val_loss = current_val_loss
        early_stop_counter = 0
        # Save best model
        paddle.save(model.state_dict(), 'output_models/best_vgg_asdid.pdparams')
    else:
        early_stop_counter += 1
        print(f'EarlyStopping counter: {early_stop_counter}/{early_stop_patience}')
        if early_stop_counter >= early_stop_patience:
            print(f'Early stopping triggered at epoch {epoch}!')
            break

metrics = pd.DataFrame({
    'epoch': range(len(trainLoss)),
    'train_loss': trainLoss,
    'train_acc': trainAcc,
    'val_loss': valLoss,
    'val_acc': valAcc

})
metrics.to_csv('results/best_vgg_asdid.csv', index=False)

# free unused memory
del train_dataset
del val_dataset
del train_loader
del val_loader
gc.collect()

train_time = time.time()
print("Total time for training: ", train_time - start_time)


def plot_training_metrics(df):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot Loss
    ax1.plot(df['epoch'], df['train_loss'], 'g-', label='Training Loss')
    ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    ax2.plot(df['epoch'], df['train_acc'], 'g-', label='Training Accuracy')
    ax2.plot(df['epoch'], df['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Calculate and print some statistics
    print("\nTraining Statistics:")
    print("-" * 50)
    print(f"Final Training Loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"Final Validation Loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"Final Training Accuracy: {df['train_acc'].iloc[-1]:.4f}")
    print(f"Final Validation Accuracy: {df['val_acc'].iloc[-1]:.4f}")
    print(f"Best Training Accuracy: {df['train_acc'].max():.4f}")
    print(f"Best Validation Accuracy: {df['val_acc'].max():.4f}")

    # Save the plot
    plt.savefig('results/metrics_vgg_asdid.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'training_test_healthy.png'")

    # Display the plot
    plt.show()


save_path = 'output_models/model_vgg_asdid.pdparams'
print('save model to: ' + save_path)
paddle.save(model.state_dict(), save_path)

end_time = time.time()
print("Total time for training and inference: ", end_time - start_time)