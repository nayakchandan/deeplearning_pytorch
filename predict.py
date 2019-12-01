import pandas as pd
import numpy as np
import torch 
import argparse
import json
from helper_functions import load_checkpoint, process_image

parser = argparse.ArgumentParser(description="parses terminal arguments predicting images")

parser.add_argument('image_path',
                    action='store',
                    help='Path of the image to be predicted')

#optional parser arguments
parser.add_argument('--check_point',
                    action='store',default='checkpoint.pth',
                    help='Loads pretrained model for inference. Default is checkpoint.pth')
parser.add_argument('--top_k',
                    action='store',type=int,default=1,
                    help='Top k results sorted in descending order by probability')                
parser.add_argument('--gpu',
                    action='store',default=False,type=bool,
                    help='True:use GPU;False:use CPU')
parser.add_argument('--dictionary',
                    action='store',default='cat_to_name.json',dest='dict_json',
                    help='Mapping to convert a label to its real world name')

inputs = parser.parse_args()

img_path = inputs.image_path
checkpoint = inputs.check_point
top_k = inputs.top_k
use_gpu = inputs.gpu
dictionary = inputs.dict_json

def main(image_path,checkpoint,top_k=1,use_gpu=False,dictionary='cat_to_name.json'):

    with open(dictionary, 'r') as f:
        cat_to_name = json.load(f)

    if use_gpu==True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor.to(device)
    
    model.to(device)
    model.eval() #set to eval mode
    
    with torch.no_grad():  # turn off autograd calcs to speed up stuff
        output = model.forward(img_tensor)
        ps = torch.exp(output) #we used logsoftmax so need to use exponent to get probabilites
        top_preds, top_labs = ps.topk(top_k)
        top_labs = top_labs.tolist()
        top_preds = top_preds.detach().numpy().tolist()
        labels = pd.DataFrame({'class':pd.Series(model.class_to_idx),'flower_name':pd.Series(cat_to_name)})
        labels = labels.set_index('class')
    
        # Limit the dataframe to top labels and add their predictions
        labels = labels.iloc[top_labs[0]]
        labels['predictions'] = top_preds[0]
    
    print("Categories selected for top "+str(top_k)+" categories are:")
    print(labels)

if __name__ == '__main__':
    print("Checking neural network...")
    model = load_checkpoint(torch.load(checkpoint))
    print("Checkpoint loaded")
    main(img_path,model,top_k,use_gpu,dictionary)  