
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import io

# ## Helper functions
# Set of helper functions

# In[3]:


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


# ## Configuration Class
# A simple class to manage configuration

# In[4]:


class Config():
    training_dir = "./data/only_traffic_light/training/"
    testing_dir = "./data/only_traffic_light/testing/"
    train_batch_size = 64  # (batch size)
    train_number_epochs = 30  # (No of epochs)
    img_res = 30  # (define the image resolution for siamese network)


# ## Custom Dataset Class
# This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair

# In[5]:


class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        ##############################3333
        # imgk=Image.open(io.BytesIO(img0_tuple[0]))
        # img0 = list(img0.getdata(0))
        # img1 = list(img1.getdata(0))
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")
        ######################################3
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")
        # list(cool.getdata(0))
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            #
            # img0 = list(img0.getdata(0))
            # img1 = list(img1.getdata(0))
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# ## Using Image Folder Dataset

folder_dataset = dset.ImageFolder(root=Config.training_dir)


# In[7]:
# res=30

siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((Config.img_res,Config.img_res)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)


# ## Visualising some of the data
# The top row and the bottom row of any column is one pair. The 0s and 1s correspond to the column of the image.
# 1 indiciates dissimilar, and 0 indicates similar.




vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=8)
dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())


# ## Neural Net Definition
# We will use a standard convolutional neural network

# In[9]:


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*Config.img_res*Config.img_res, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# ## Contrastive Loss

# In[10]:


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
         = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


# ## Training Time!

# In[22]:
train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=Config.train_batch_size)


# In[23]:


net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )


# In[24]:


counter = []
loss_history = [] 
iteration_number= 0


# In[25]:


for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        # t1 = time.time()
        output1,output2 = net(img0,img1)
        # print("train time taken", (time.time() - t1) * 1000)

        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
torch.save(net.state_dict(), './model-save_dict_gwm_tl-%s.pt' % epoch)
show_plot(counter,loss_history)

###########################33
# torch.save(net.state_dict(), './model-epoch-%s.pth' % epoch)
# torch.save(net, './model-epoch_traffic_light_new-%s.pt' % epoch)
# torch.save(net.state_dict(), './model-epoch-%s.pth' % epoch)
# net.load_state_dict(saved_model)
########################33333
#diff way of storing
# torch.save(net.state_dict(), './model-save_dict_osl_all_gwm_tl-%s.pt' % epoch)
################################
# ## Some simple testing
# The last 3 subjects were held out from the training, and will be used to test. The Distance between each image pair denotes the degree of similarity the model found between the two images. Less means it found more similar, while higher values indicate it found them to be dissimilar.


#this is test datasets

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((Config.img_res,Config.img_res)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)

for i in range(20):
    _,x1,label2 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    t1=time.time()
    output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
    print("actual time taken", (time.time()-t1)*1000)
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))




