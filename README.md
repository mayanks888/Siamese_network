# Image Similarity with Siamese Networks in Pytorch
You can read the accompanying article at https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e

The goal is to teach a siamese network to be able to distinguish pairs of images. 
This project uses pytorch. 

Any dataset can be used. Each class must be in its own folder. This is the same structure that PyTorch's own image folder dataset uses.



## Installing the right version of PyTorch 
This project is updated to be compatible with pytorch 0.4.0 and requires python 3.5


You can find other project requirements in `requirements.txt` , which you can install using `pip install -r requirements.txt`



## Training model 

    class Config()
        training_dir = "./data/only_traffic_light/training/"
        testing_dir = "./data/only_traffic_light/testing/"
        train_batch_size = 64 #(batch size)
        train_number_epochs = 30 #(No of epochs)
        img_res=30 #(define the image resolution for siamese network)
       
     Run : Train.py
     
## Inference 
    Class Config()
        training_dir = "./data/only_traffic_light/training/"
        testing_dir = "./data/only_traffic_light/testing/"
        train_batch_size = 64 #(batch size)
        train_number_epochs = 30 #(No of epochs)
        img_res=30 #(define the image resolution for siamese network)
       
     Run : Inference.py