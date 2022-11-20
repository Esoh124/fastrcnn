import torch

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
import os
import random
import numpy as np
import shutil
import cv2

#print(len(os.listdir('annotations')))
#print(len(os.listdir('images')))

#random.seed(1234)
#idx = random.sample(range(853), 170)

#for img in np.array(sorted(os.listdir('images')))[idx]:
#    shutil.move('images/'+img, 'test_images/'+img)

#for annot in np.array(sorted(os.listdir('annotations')))[idx]:
#    shutil.move('annotations/'+annot, 'test_annotations/'+annot)

#print(len(os.listdir('annotations')))
#print(len(os.listdir('images')))
#print(len(os.listdir('test_annotations')))
#print(len(os.listdir('test_images')))

import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from PIL import Image
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
def unconvert(class_id, width, height, x, y, w, h):
    
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)

def generate_box(obj, file):
    img = cv2.imread(file)
    height, width, channels = img.shape
    obj = [float(s) for s in obj.split()]
    #print(obj)
    classes = [0, 1]
    new_label = unconvert(obj[0], width, height, obj[1], obj[2], obj[3], obj[4])
    label = classes[new_label[0]]
    xmin = float(new_label[1])
    ymin = float(new_label[3])
    xmax = float(new_label[2])
    ymax = float(new_label[4])
    
    return [label, xmin, ymin, xmax, ymax]

adjust_label = 1

def generate_label(obj):

    if obj.find('name').text == "with_mask":

        return 1 + adjust_label

    elif obj.find('name').text == "mask_weared_incorrect":

        return 2 + adjust_label

    return 0 + adjust_label

def generate_target(file, imgfile): 
    with open(file) as f:
        data = f.readlines()
        #soup = BeautifulSoup(data, "html.parser")
        #objects = soup.find_all("object")

        num_objs = len(data)
        
        boxes = []
        labels = []
        box=[]
        for i in data:#a line label x y w h
            #print(i)
            box = generate_box(i, imgfile)#label, xmin ymin xmax ymax
            #print(box[1:])
            boxes.append(box[1:])
            labels.append(box[0])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) 
        #print(box.is_cuda)
        labels = torch.as_tensor(labels, dtype=torch.int64) 
        #print(labels)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        return target

def plot_image_from_output(img, annotation):
    
    img = img.cpu().permute(1,2,0)
    
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    
    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 1 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        
        elif annotation['labels'][idx] == 2 :
            
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
            
        else :
        
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')

        ax.add_patch(rect)

    plt.show()
    
    
class MaskDataset(object):
    def __init__(self, transforms, path):
        '''
        path: path to train folder or test folder
        '''
        # transform module과 img path 경로를 정의
        self.transforms = transforms
        self.path = path
        
        self.all_imgs = list(sorted(os.listdir(os.path.join(path, "images"))))
        self.all_dicts = list(sorted(os.listdir(os.path.join(path, "labels"))))
        #self.imgs = list(sorted(os.listdir(self.path)))


    def __getitem__(self, idx): #special method
        # load images ad masks
        #file_image = self.all_imgs[idx]
        #file_label = self.all_dicts[idx]
        img_path = os.path.join(self.path,"images",self.all_imgs[idx])
        label_path = os.path.join(self.path,"labels",self.all_dicts[idx])
        
        
        #label_path = os.path.join("annotations/", self.all_dicts)

        img = Image.open(img_path).convert("RGB")
        
        #Generate Label
        target = generate_target(label_path, img_path)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self): 
        return len(self.all_imgs)

data_transform = transforms.Compose([  # transforms.Compose : list 내의 작업을 연달아 할 수 있게 호출하는 클래스
        transforms.ToTensor() # ToTensor : numpy 이미지에서 torch 이미지로 변경
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

dataset = MaskDataset(data_transform, './train')
test_dataset = MaskDataset(data_transform, './test')

data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)

num_classes = 3 #1,5 back

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
#replace model head with new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
torch.cuda.empty_cache()
model.to(device)
    
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
  
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 50
print('----------------------train start--------------------------')
for epoch in range(num_epochs):
    start = time.time()
    model.train()
    i = 0    
    epoch_loss = 0
    for im, annotations in data_loader:
        i += 1
        im = list(img.to(device) for img in im)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        
        
        #print(annotations)
        #im = im.cuda()
        #im = [i.cuda() for i in im]
        #annotations = [a.cuda() for a in annotations]
    
        #annotations = annotations.cuda()
        loss_dict = model(im, annotations) 
        losses = sum(loss for loss in loss_dict.values())        

        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
        epoch_loss += losses
    print(f'epoch : {epoch+1}, Loss : {epoch_loss}, time : {time.time() - start}')
    model.eval()

    
torch.save(model.state_dict(),f'model_{num_epochs}.pt')
model.load_state_dict(torch.load(f'model_{num_epochs}.pt'))

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : 
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

with torch.no_grad(): 
    for imgs, annotations in test_data_loader:
        imgs = list(img.to(device) for img in imgs)
        pred = make_prediction(model, imgs, 0.5)
        print(pred)
        break
    
    
_idx = 1
print("Target : ", annotations[_idx]['labels'])
plot_image_from_output(imgs[_idx], annotations[_idx])
print("Prediction : ", pred[_idx]['labels'])
plot_image_from_output(imgs[_idx], pred[_idx])