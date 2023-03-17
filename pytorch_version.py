#hide
# ! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

#hide
from fastai.vision.all import *
from fastbook import *


from fastai.vision.all import *
import torchvision
import torchvision.transforms as transforms
# from livelossplot import PlotLosses

URLs.MNIST

path = untar_data(URLs.MNIST, dest="/workspace/data")
Path.BASE_PATH = path
path.ls()

(path/"training").ls()

(path/"training/1").ls()


image = Image.open((path/"training/1").ls()[0])
image

image.size
image.mode


transform = transforms.Compose(
    [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)

# Above are the transformations we will make to each of the images when creating our Pytorch datasets.

# Step 1: Converting into a grayscale image, i.e. fusing the RGB color channels into a grayscale one (from what would be a [3, 28, 28] tensor to a [1, 28, 28]).

#  Tip: We need to do this because the loader parameter of ImageFolder (see next cell) loads 3 channels even if the original image only has one. I couldn’t bother creating a custom loader so this does the trick.
# Step 2: Converting the grayscale image (with pixel values in the range [0, 255] into a 3 dimensional [1, 28, 28] pytorch tensor (with values in the range [0, 1]).

# Step 3: We normalize with mean = 0.5 and std = 0.5 to get values from pixels in the range [-1, 1]. (pixel = (image - mean) / std maps 0 to -1 and 1 to 1).

#  Note: The argument of centering around 0 is usually held for activation functions inside the network (which we aren’t doing because we are using ReLU) but I did it for the input layer because I felt like it. This is not the same as standardizing but still gives a zero centered range. You can read this and the links given for more info.

full_dataset = torchvision.datasets.ImageFolder((path/"training").as_posix(), transform = transform)

# Splitting the above dataset into a training and validation dataset
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
training_set, validation_set = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

# Dataset using the "testing" folder
testing_set = torchvision.datasets.ImageFolder((path/"testing").as_posix(), transform = transform)

bs = 64

train_loader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=bs)
dataloaders = {
    "train": train_loader,
    "validation": validation_loader
}
# Original architecture
# pytorch_net = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(28*28, 128),
#     nn.ReLU(),
#     nn.Linear(128, 50),
#     nn.ReLU(),
#     nn.Linear(50,10),
#     nn.LogSoftmax(dim=1))

#I added a third layer for experimentation
pytorch_net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 50),
    nn.ReLU(),
    nn.Linear(50,30),
    nn.ReLU(),
    nn.Linear(30,10),
    nn.LogSoftmax(dim=1))
# Wanna know why logsoftmax instead of softmax only ? https://deepdatascience.wordpress.com/2020/02/27/log-softmax-vs-softmax/

# Before moving on, we should define a bunch of variables. "A torch.device is an object representing the device on which a torch.Tensor is or will be allocated". Head here for more info. Here we want to perform our computations on a GPU if it is available.

# lr is our learning rate hyperparameter representing the size of the step we take when applying SGD.

# nb_epoch is our number of epochs, meaning the number of complete passes through the training dataset.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr = 1e-2
nb_epoch = 77

optimizer = torch.optim.SGD(pytorch_net.parameters(), lr=lr)

criterion = nn.NLLLoss()

# We chose pytorch's nn.NLLLoss() for our loss function. It stands for negative log likelihood loss and is useful to train a classification problem with more than 2 classes. It expects log-probabilities as input for each class, which is our case after applying LogSoftmax.

# Tip: Instead of applying a LogSoftmax layer in the last layer of our network and using NLLLoss, we could have used CrossEntropyLoss instead which is a loss that combines the two into one single class. Read the doc[https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html] for more.

def train_model(model, criterion, optimizer, dataloaders, num_epochs=10):
    liveloss = PlotLosses() # Live training plot generic API
    model = model.to(device) # Moves and/or casts the parameters and buffers to device.
    
    for epoch in range(num_epochs): # Number of passes through the entire training & validation datasets
        logs = {}
        for phase in ['train', 'validation']: # First train, then validate
            if phase == 'train':
                model.train() # Set the module in training mode
            else:
                model.eval() # Set the module in evaluation mode

            running_loss = 0.0 # keep track of loss
            running_corrects = 0 # count of carrectly classified inputs

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) # Perform Tensor device conversion
                labels = labels.to(device)

                outputs = model(inputs) # forward pass through network
                loss = criterion(outputs, labels) # Calculate loss

                if phase == 'train':
                    optimizer.zero_grad() # Set all previously calculated gradients to 0
                    loss.backward() # Calculate gradients
                    optimizer.step() # Step on the weights using those gradient w -=  gradient(w) * lr

                _, preds = torch.max(outputs, 1) # Get model's predictions
                running_loss += loss.detach() * inputs.size(0) # multiply mean loss by the number of elements
                running_corrects += torch.sum(preds == labels.data) # add number of correct predictions to total

            epoch_loss = running_loss / len(dataloaders[phase].dataset) # get the "mean" loss for the epoch
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset) # Get proportion of correct predictions
            
            # Logging
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'

            logs[prefix + 'log loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc.item()
        
        liveloss.update(logs) # Update logs
        liveloss.send() # draw, display stuff

train_model(pytorch_net, criterion, optimizer, dataloaders, nb_epoch)

#  toch.save(pytorch_net, 'models/pytorch-97.7acc.pt')
