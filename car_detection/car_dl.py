import os
import random
import torch
import torchvision
from torchvision import transforms as Tr
from torch.cuda.random import seed
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.io import read_image
from PIL import Image , ImageDraw
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision .models .detection. faster_rcnn import FastRCNNPredictor
torch.manual_seed(42)

print(torch.cuda.is_available())
print(torch.cuda.device_count())

#Reading the csv file
bd = pd.read_csv('path to csv file')
bd.head()

#Getting unique images from the training images and csv file.
img_pth = 'path to training images'
img_unq = bd.image.unique()
print("The number of unique images is:",len(img_unq))

#Creating DLCarData using the training_images
class DLCarData(torch.utils.data.Dataset):
    def __init__(self, bd, img_unq, indcs):
        self.bd = bd
        self.img_unq = img_unq
        self.indices = indcs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img_name = self.img_unq[self.indices[index]]
        bbx = self.bd[self.bd.image == img_name].values[: , 1:].astype("float")
        img = Image.open(img_pth + img_name).convert('RGB')
        labels = torch.ones((bbx.shape[0]) , dtype = torch.int64)
        targt = {}
        targt["boxes"] = torch.tensor(bbx)
        targt["label"] = labels

        return Tr.ToTensor()(img) , targt
    
#Splitting the DLCarData into Training dataset and Validation Dataset    
tr_indcs , vl_indcs = train_test_split(range(img_unq.shape[0]) , test_size = 0.4)

def collate(data):
    return data

tr_dlr = torch.utils.data.DataLoader(DLCarData(bd , img_unq, tr_indcs), 
                                       batch_size = 4,
                                       shuffle = True,
                                       collate_fn = collate,
                                       pin_memory = True if torch.cuda.is_available() else False)
                                       
vl_dlr = torch.utils.data.DataLoader(DLCarData(bd , img_unq, vl_indcs), 
                                       batch_size = 2,
                                       shuffle = True,
                                       collate_fn = collate,
                                       pin_memory = True if torch.cuda.is_available() else False)


#Downloading pretrainedmodel Resnet 50 for training the custom dataset
CarModel = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
ncs = 2
n_fetures = CarModel.roi_heads.box_predictor.cls_score.in_features
CarModel.roi_heads.box_predictor = FastRCNNPredictor(n_fetures , ncs)

#Activating the GPU and using CUDA
dvce = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(dvce)

#Setting ther optimizer value
optmzr = torch.optim.SGD(CarModel.parameters() , lr=0.001, momentum=0.9, weight_decay=0.0005)

#Training the DLCarData Model
ne = 50
CarModel.to(dvce)
for epochs in range(ne):
    epoch_loss = 0
    progress_bar = tqdm(tr_dlr, desc=f'Epoch {epochs + 1}/{ne}')
    for data in progress_bar:
        img = []
        targets = []
        for d in data:
            img.append(d[0].to(dvce))
            targt = {}
            targt["boxes"] = d[1]["boxes"].to(dvce)
            targt["labels"] = d[1]["label"].to(dvce)
            targets.append(targt)
        ls_dict = CarModel(img , targets)
        ls = sum(v for v in ls_dict.values())
        epoch_loss += ls.cpu().detach().numpy()
        optmzr.zero_grad()
        ls.backward()
        optmzr.step()
       
    print(f'Loss = {epoch_loss}')

#Saving the trained model in the Working Directory
torch.save(CarModel.state_dict(), 'CarModel.pt')

#Displaying the Validation images
CarModel.eval()

dt = iter(vl_dlr).__next__()
img = dt[0][0]
bbxs = dt[0][1]["boxes"]
lbls = dt[0][1]["label"]
rslt = CarModel([img.to(dvce)])
rslt_bbx = rslt[0]["boxes"]
rslt_scrs = rslt[0]["scores"]
kep = torchvision.ops.nms(rslt_bbx , rslt_scrs, 0.45)
rslt_bbx.shape , kep.shape
imn =(img.permute(1,2,0).cpu().detach().numpy()*255).astype('uint8')
rsmple = Image.fromarray(imn)
draw = ImageDraw.Draw(rsmple)
for box in bbxs:
    draw.rectangle(list(box) , fill = None , outline = 'cyan')
rsmple

CarModel.eval()

tcorrect = 0
tsamples = 0

with torch.no_grad():
    for data in vl_dlr:
        imgs = []
        targets = []
        for d in data:
            imgs.append(d[0].to(dvce))
            targ = {}
            targ["boxes"] = d[1]["boxes"].to(dvce)
            targ["labels"] = d[1]["label"].to(dvce)
            targets.append(targ)
        
        opt = CarModel(imgs)
        for i in range(len(opt)):
            pred_boxes = opt[i]['boxes']
            pred_labels = opt[i]['labels']
            
            # perform non-maximum suppression to get the final predicted boxes
            keep = torchvision.ops.nms(pred_boxes, pred_labels, iou_threshold=0.5)
            pred = pred_boxes[keep]
            pred = pred_labels[keep]

            # count the number of correct predictions
            for j in range(len(pred_boxes)):
                tboxes = targets[i]['boxes']
                tlabels = targets[i]['labels']
                tboxes = tboxes[tlabels == 1]
                iou = torchvision.ops.box_iou(pred_boxes[j].unsqueeze(0), tboxes)
                if iou.max() > 0.815:
                    tcorrect += 1
        
        tsamples += len(data)

valid_acc = tcorrect / tsamples
print("The Validation accuracy is = {:.2f}%".format(valid_acc * 100))


# Loading the trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
model.load_state_dict(torch.load('CarModel.pt'))
model.eval()

# Defining the path to the images
testingpath = 'path to test images'
names = os.listdir(testingpath)


all_boxes = []


iou_threshold = 0.2
print("Predicting Boxes of the Test Data")

for name in tqdm(names):
    
    image = Image.open(os.path.join(testingpath, name)).convert('RGB')
    
    #Predict the car in the images
    with torch.no_grad():
        pred = model([Tr.ToTensor()(image)])
    
    #Predict boxes and labels
    boxes = pred[0]['boxes'].detach().numpy()
    scores = pred[0]['scores'].detach().numpy()
    labels = pred[0]['labels'].detach().numpy()
    
    
    kp_indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)
    boxes = boxes[kp_indices]
    labels = labels[kp_indices]
    
    # Add boxes and label to list
    all_boxes.append({'image_name': name, 'boxes': boxes, 'labels': labels})