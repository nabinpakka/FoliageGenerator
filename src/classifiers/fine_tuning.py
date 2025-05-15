import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.metric import Precision, Recall
from paddle.vision.models.vgg import make_layers
from swin_transformer import SwinTransformer

import numpy as np

import os

# Assuming the ConvNeXt with CBAM code you provided is in a file called 'model.py'
from conv_next import ConvNeXt
from dataset_station import StationType

paddle.set_device('gpu:1')

# datasets
train_dataset = StationType(mode='val')
val_dataset = StationType(mode='test')
test_dataset = StationType(mode='train')

trained_model_path = "path_to_pre_trained_model"


def evaluate_model(model, test_loader, num_classes):
    model.eval()
    test_correct = 0
    test_total = 0

    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Initialize paddle metrics
    precision_metric = Precision()
    recall_metric = Recall()

    with paddle.no_grad():
        for images, labels in test_loader():
            outputs = model(images)
            pred = outputs.argmax(axis=1)

            # Calculate accuracy
            test_correct += (pred == labels.squeeze()).astype(paddle.float32).sum().item()
            test_total += labels.shape[0]

            # Update metrics
            precision_metric.update(pred, labels.squeeze().reshape([-1]))
            recall_metric.update(pred, labels.squeeze().reshape([-1]))

            # Update confusion matrix
            for i in range(len(pred)):
                confusion_matrix[labels.squeeze().reshape([-1])[i].item()][pred[i].item()] += 1

    # Calculate metrics for each class and average
    test_acc = test_correct / test_total

    # Get precision and recall from paddle metrics
    precision = precision_metric.accumulate()
    recall = recall_metric.accumulate()

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # add small epsilon to avoid division by zero

    # Calculate metrics manually from confusion matrix if needed
    precision_manual = np.zeros(num_classes)
    recall_manual = np.zeros(num_classes)
    f1_manual = np.zeros(num_classes)

    for i in range(num_classes):
        # True positives: diagonal elements
        tp = confusion_matrix[i, i]
        # False positives: sum of column i minus true positive
        fp = np.sum(confusion_matrix[:, i]) - tp
        # False negatives: sum of row i minus true positive
        fn = np.sum(confusion_matrix[i, :]) - tp

        # Calculate precision, recall, and F1 for each class
        precision_manual[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_manual[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_manual[i] = 2 * precision_manual[i] * recall_manual[i] / (precision_manual[i] + recall_manual[i]) if (precision_manual[i] +recall_manual[i]) > 0 else 0

    # Calculate macro averages
    macro_precision = np.mean(precision_manual)
    macro_recall = np.mean(recall_manual)
    macro_f1 = np.mean(f1_manual)

    # Print results
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("Confusion matrix: ", confusion_matrix)

    print("\nPer-class metrics (manually calculated):")
    for i in range(num_classes):
        print(
            f"Class {i} - Precision: {precision_manual[i]:.4f}, Recall: {recall_manual[i]:.4f}, F1: {f1_manual[i]:.4f}")

    print(f"\nMacro-average - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")

    return {
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix
    }


# Usage example:
# metrics = evaluate_model(model, test_loader, num_classes=3)

def apply_transfer_learning():
    # 1. Load pre-trained model
    # Create model with original number of classes (e.g., ImageNet has 1000)
    # model = ConvNeXt(in_chans=3, num_classes=10)

    # DenseNet121 = paddle.vision.models.DenseNet(layers=121,num_classes=10)
    # MobileNetV3 = paddle.vision.models.MobileNetV3Large(scale=1.0, num_classes=3)
    # ShuffleNetV2 = paddle.vision.models.ShuffleNetV2(scale=1.0, act='relu', num_classes=3)
    # SqueezeNet = paddle.vision.models.SqueezeNet(version = '1.0', num_classes=3)

    # vgg19_cfg = [64,64, 'M', 128,128, 'M', 256, 256,256, 256,'M', 512, 512,512,512, 'M', 512, 512,512,512, 'M']
    # features = make_layers(vgg19_cfg)
    # VGG = paddle.vision.models.VGG(features, num_classes=10)
    # model=  VGG

    ResNet50 = paddle.vision.models.resnet50(num_classes=10, pretrained=False)
    model = ResNet50

    # model=SwinTransformer(num_classes=10)

    # Load pre-trained weights if available
    if os.path.isfile(trained_model_path):
        param_state_dict = paddle.load(trained_model_path)
        model.set_state_dict(param_state_dict)
        print("Pre-trained weights loaded successfully!")
    else:
        print("No pre-trained weights found, using model with random initialization")

    # 2. Modify the model for new task (e.g., for a classification task with 10 classes)
    target_num_classes = 3  # Change this to match your task

    # # Store the original weights of everything except the head
    # original_params = {}
    # for name, param in model.named_parameters():
    #     if 'head' not in name:
    #         original_params[name] = param

    # # Replace head with a new head for the target task
    # model.head = nn.Linear(model.head.weight.shape[0], target_num_classes)

    # # 3. Freeze the early layers (e.g., first two stages)
    # for name, param in model.named_parameters():
    #     if 'stages.0' in name or 'stages.1' in name or 'downsample_layers.0' in name or 'downsample_layers.1' in name:
    #         param.trainable = False

    # Replace the final classifier layer (for Densenet)
    # num_fltr = model.classifier[-1].weight.shape[1]
    # model.classifier[-1] = paddle.nn.Linear(num_fltr, target_num_classes)

    # Replace the final classifier layer (for ResNet50)
    num_fltr = model.fc.weight.shape[0]
    model.fc = paddle.nn.Linear(num_fltr, target_num_classes)

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=False)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=4, shuffle=True, drop_last=False)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=False)

    # 5. Set up optimizer and loss
    # Only train unfrozen parameters
    trainable_params = filter(lambda p: p.trainable, model.parameters())
    optimizer = paddle.optimizer.Adam(parameters=trainable_params, learning_rate=0.0001)
    criterion = paddle.nn.loss.CrossEntropyLoss()

    # 6. Training loop
    num_epochs = 100

    early_stop_patience = 5
    early_stop_delta = 0.01
    best_val_loss = np.inf
    early_stop_counter = 0

    trainLoss, valLoss = [], []
    trainAcc, valAcc = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_id, data in enumerate(train_loader()):
            images, labels = data

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            total_loss += loss.item()

            # Calculate accuracy
            pred = outputs.argmax(axis=1)
            correct += (pred == labels.squeeze()).sum().item()
            total += labels.shape[0]

            if batch_id % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_id}], "
                      f"Loss: {loss.item():.4f}, Accuracy: {correct / total:.4f}")

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

        trainAcc.append(avg_acc)

        # 7. Evaluation
        model.eval()

        listValLoss = []
        listValAcc = []
        with paddle.no_grad():
            for images, val_y in val_loader():
                y_pred = model(images)
                loss = criterion(y_pred, val_y)
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
            paddle.save(model.state_dict(), 'output_models/tomato/5/best_finetuned_natural_9.pdparams')
        else:
            early_stop_counter += 1
            print(f'EarlyStopping counter: {early_stop_counter}/{early_stop_patience}')
            if early_stop_counter >= early_stop_patience:
                print(f'Early stopping triggered at epoch {epoch}!')
                break

        # 8. Progressive unfreezing (after some epochs)
        if epoch == 4:  # Unfreeze the second stage after 5 epochs
            for name, param in model.named_parameters():
                if 'stages.1' in name:
                    param.trainable = True

            # Update optimizer with newly unfrozen parameters
            trainable_params = filter(lambda p: p.trainable, model.parameters())
            optimizer = paddle.optimizer.Adam(parameters=trainable_params, learning_rate=0.00005)
            print("Unfrozen stage 2 layers and reduced learning rate")

    evaluate_model(model, test_loader, 3)

    return model


# Execute transfer learning
if __name__ == "__main__":
    model = apply_transfer_learning()