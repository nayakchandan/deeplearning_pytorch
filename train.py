import argparse
from helper_functions import data_loader,create_model,train_nn,save_checkpoint
from torchvision import datasets, transforms, models

# setup parsers to read user input in terminal and translate to parameters
parser = argparse.ArgumentParser(description="parses terminal arguments for training Neural Network")
parser.add_argument('data_dir',
                    action='store',help='Path to cataloged training set')
# optional parser arguments
parser.add_argument('--save_dir',
                    action='store',default='checkpoint.pth',help='saves checkpoint to a this location')
parser.add_argument('--learn_rate',
                    action='store',dest='lr',type=float,default=0.001,
                    help='Set the learning rate, default is 0.001')
parser.add_argument('--dropout',
                    action='store',dest='dropout',type=float,default=0.5,
                    help='sets the dropout parameter for classifier, default is 0.5')
parser.add_argument('--hidden_units',
                    action='store',dest='units',type=int,default=512,
                    help='number of hidden units in the classifier neural network')
parser.add_argument('--epochs',
                    action='store',dest='num_epochs',type=int,default=2,
                    help='number of epochs for training, default is 2')
parser.add_argument('--gpu',
                    action='store',default=False,type=bool,
                    help='True:uses GPU;False:uses CPU')
parser.add_argument('--batch_size',
                    action='store',default=32,dest='batch_size',type=int,
                    help='Sets batch size for loading images from training directory, default is 32')
parser.add_argument('--nn_model',
                    action='store',default='vgg',dest='model',choices=['vgg','densenet'],
                    help='Choose neural network classifier using keywords vgg for VGG16 ,densenet for Densenet161')

inputs = parser.parse_args()

data_dir = inputs.data_dir
save_dir = inputs.save_dir
learn_rate = inputs.lr
dropout = inputs.dropout
hidden_units = inputs.units
epochs = inputs.num_epochs
gpu_mode = inputs.gpu
batch_size = inputs.batch_size
nn_model = inputs.model

def main(data_dir,save_dir,learn_rate,dropout,hidden_units,epochs,gpu_mode,batch_size):    
    print('----MAIN MODULE v1.0----')
    print("Main function reporting, received inputs, preparing training data...")
    trainloader, validloader, train_data, valid_data = data_loader(data_dir,batch_size)
    outputs = len(train_data.classes) # this will provide the number of categories for classification. Used to create the classifier neural network
    print("Created data loader!")
    print("Creating neural network with user provided inputs...")
    model = create_model(outputs,hidden_units,dropout,nn_model)
    print("Neural network created! Proceeding to training...")
    train_nn(trainloader,validloader,model,epochs,learn_rate,gpu_mode)
    print("Network is now trained on input data")
    save_checkpoint(model,train_data,save_dir)
    print("Model has been saved sucessfully")
    print("Complete!")


if __name__ == '__main__':

    print("Parsing inputs...")
    print("Sent to main function for further processing...")
    main(data_dir,save_dir,learn_rate,dropout,hidden_units,epochs,gpu_mode,batch_size)
    print("Exiting...")
