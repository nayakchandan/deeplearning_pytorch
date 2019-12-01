import numpy as np 
import pandas as pd 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def data_loader(data_dir,size=32):
    """
    This function creates the transformations and generator for loading training images
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(
                                    train_data, batch_size=size, shuffle=True)
    validloader = torch.utils.data.DataLoader(
                                    valid_data, batch_size=size)   
    
    return trainloader, validloader, train_data, valid_data

def create_model(output_layer,hidden_units=512,dropout=0.5,model_typ='vgg'):
    """
    Creates a VGG16 neural network, freezes learning in features set
    and creates a simple classfier with hidden units and dropout specified by user 
    """
    if model_typ == 'vgg':
        train_model = models.vgg16(pretrained=True)
        input_units = train_model.classifier[0].in_features
    elif model_typ == 'densenet':
        train_model = models.densenet161(pretrained=True)
        input_units = train_model.classifier.in_features
    
    for param in train_model.parameters():
        param.requires_grad = False
        
    
    classifier = nn.Sequential(nn.Linear(input_units,hidden_units,bias=True),
                           nn.ReLU(),
                           nn.Dropout(dropout),
                           nn.Linear(hidden_units,output_layer,bias=True),
                           nn.LogSoftmax(dim=1))

    train_model.classifier = classifier
    print('Model created with following parameters:')
    print('Activation function: ReLU')
    print('Input units: '+str(input_units))
    print('Hidden units: '+str(hidden_units))
    print('Output units or number of objects being classified: '+str(output_layer))
    print('Dropout: '+str(dropout))
    return train_model

def train_nn(trainloader,validloader,model,epochs,learning_rate=0.001,gpu=False):
    """
    train the neural network on the training set 
    """

    ep = epochs
    steps = 0
    print_every=50
    running_loss=0
    if gpu==True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
    model.train()
    for e in range(ep):
        for inputs,labels in trainloader:
            steps+=1
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps%print_every==0:
                val_loss = 0
                accuracy = 0
                model.eval() #switch model to evaluation mode to stop training
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs,labels = inputs.to(device),labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps,labels)
                        val_loss += batch_loss.item()
                        # accuracy
                        ps = torch.exp(logps) # convert to probabilites
                        top_p,top_class = ps.topk(1,dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        #print(len(validloader))
                print(f"Epoch {e+1}/{ep}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {val_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train() # set model to train mode again
    print("Training Complete!")
    return model

def save_checkpoint(model,train_data,path='checkpoint.pth'):
    """
    saves trained model parameters to a checkpoint defined by user
    """
    model.class_to_idx = train_data.class_to_idx
    checkpoint = OrderedDict()
    checkpoint = {'classifier':model.classifier,
                  'classifier': model.classifier,
                  'class_to_idx':model.class_to_idx,
                  'state_dict':model.state_dict(),
                  'model_key':str(type(model))}
    torch.save(checkpoint,path)

def load_checkpoint(checkpoint):
    '''
    Input - checkpoint of a vgg16 model
    Output - model restored with checkpoint parameters
    '''
    
    # create a vgg16 model and stop training on the features component
    model_key = checkpoint['model_key']
    if 'VGG' in model_key:
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet161(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''       
    #open image and find out size
    im = Image.open(image)
    width, height = im.size
    #resize lowest dimension to 256 and scale the other side appropriately to maintain original aspect ratio
    if width<height:
        scale = 256/width
        height = int(height*scale)
        width = 256
    else:
        scale = 256/height
        width = int(width*scale)
        height = 256

    im.thumbnail((width,height)) #we maintain aspect ratio
    
    # centre crop
    # PIL has a weird co-ordinate system, so need to follow it, PIL documentation is garbage
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    # Crop the center of the image
    im_crop = im.crop((left, top, right, bottom))
    
    # send cropped stuff to numpy
    
    np_image = np.array(im_crop)/255
    
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image-means)/std # numpy vectorization :)
    np_image = np_image.transpose(2, 0, 1) # rearrange 
    
    return np_image