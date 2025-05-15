import numpy as np
import matplotlib.pyplot as plt

import paddle as paddle
from paddle.hapi.dynamic_flops import flops
from paddle.vision.models.vgg import make_layers

from dataset_station import StationType
from conv_next import ConvNeXt
from swin_transformer import SwinTransformer

from sklearn.metrics import confusion_matrix,f1_score,classification_report
from sklearn.metrics import roc_curve, roc_auc_score,auc
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


preds = []
labels = []
listTestLoss = []
listTestAcc = []

paddle.set_device('gpu:1')


model_state_dict = paddle.load("path_to_model")

# model=SwinTransformer(num_classes=10)

model=ConvNeXt(num_classes=10, num_patch=0)
# DenseNet121 = paddle.vision.models.DenseNet(layers=121,num_classes=10)
# MobileNetV3 = paddle.vision.models.MobileNetV3Large(scale=1.0, num_classes=3)
# ShuffleNetV2 = paddle.vision.models.ShuffleNetV2(scale=1.0, act='relu', num_classes=3)
# SqueezeNet = paddle.vision.models.SqueezeNet(version = '1.0', num_classes=10)
# vgg19_cfg = [64,64, 'M', 128,128, 'M', 256, 256,256, 256,'M', 512, 512,512,512, 'M', 512, 512,512,512, 'M']
# features = make_layers(vgg19_cfg)
# VGG = paddle.vision.models.VGG(features, num_classes=10)

# model = DenseNet121

# ResNet50 = paddle.vision.models.resnet50(num_classes=10, pretrained=False)
# model=ResNet50

# model=paddle.Model(SwinTransformer())
# model.summary((-1,3,224,224))

model.set_state_dict(model_state_dict)
model.eval()



test_dataset = StationType(mode='test')
test_loader = paddle.io.DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=True)
y_p = []
y_t = []
y_score=[]
for batch, (test_X,test_y) in enumerate(test_loader):
    y_pred = model(test_X)
    for i in y_pred:
        y_score.append(y_pred.numpy()[0])
        y_p.append(np.argmax(i))
    #y_p.append(y_pred)
    for i in test_y:
        y_t.append(int(i))
# print(y_t)
# print(y_p)


print('f1_score:',f1_score(y_t,y_p,average='macro'))
print(classification_report(y_t,y_p,digits=6))

# plt.figure(figsize=(3,3))
# # plt.rcParams['font.family']='Arial'
# # plt.rcParams['axes.unicode_minus']=False
# #confusion_matrix(y_t,y_p)
# sns.heatmap(confusion_matrix(y_t,y_p),cmap='Blues',annot=True,fmt="d")
# plt.xlabel('predict')
# plt.ylabel('real')
# plt.savefig('results/8090/confusion_cbam_512.png', dpi=300, bbox_inches='tight')
# plt.show()



