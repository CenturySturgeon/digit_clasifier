#hide
# ! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

#hide
from fastai.vision.all import *
from fastbook import *

matplotlib.rc('image', cmap='Greys')

path = untar_data(URLs.MNIST_SAMPLE)

#hide
Path.BASE_PATH = path

(path/'train').ls()

threes_tensors = [tensor(Image.open(o)) for o in (path/'train'/'3').ls().sorted()]
stacked_threes = torch.stack(threes_tensors).float()/255
stacked_threes.shape

sevens_tensors = [tensor(Image.open(o)) for o in (path/'train'/'7').ls().sorted()]
stacked_sevens = torch.stack(sevens_tensors).float()/255
stacked_sevens.shape

#reshape the tensors and concatenate them in a new one
test_tensor = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
test_tensor.shape

# We set the labels as 1 for three's images and 0 for seven's images for the test dataset
# this tensor must be the same rank (2) as the test tensor, hence the unsqueeze method
labels_test = tensor([1]*len(threes_tensors) + [0]*len(sevens_tensors)).unsqueeze(1)
labels_test.shape

# images and labels as a list of tuples (a dataset)
dset_test = list(zip(test_tensor, labels_test))
#x,y = dset_test[0]
#x.shape, y

valid3s_tensor = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()]).float()/255
valid7s_tensor = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()]).float()/255
valid_tensor = torch.cat([valid3s_tensor, valid7s_tensor]).view(-1, 28*28)
valid_tensor.shape

labels_valid = tensor([1]*len(valid3s_tensor) + [0]*len(valid7s_tensor)).unsqueeze(1)
labels_valid.shape

dset_valid = list(zip(valid_tensor, labels_valid))

# std=1 is equal to 1 because its the default standard deviation of the torch.randn function,
# but the parameter allows you to modify it. Remember, the standard deviation controls the width of the bell in the normal distribution curve
# '_' at the end of a method means to apply it in place
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()

weights = init_params((28*28, 1))
bias = init_params(1)

my_prediction = (test_tensor[0]*weights.T).sum() + bias
my_prediction

def linear1(batch): return batch@weights + bias

predictions = linear1(test_tensor)
predictions

predictions[5]

#since predictions can be <0 or >1 we apply the sigmoid function to restrict this
def sigmoid(x): return 1/(1 + torch.exp(-x))

#check for accuracy 
corrects = (predictions>0.0).float() == labels_test
corrects

accuracy = corrects.float().mean()
accuracy, accuracy.item()

#apply sigmoid to the predictions
sigmoid_pred = predictions.sigmoid()
sigmoid_pred

sigmoid_pred.shape, labels_test.shape

# For some reaseon, if I include the sqrt it fails
# Note: You shouldn't use MSE since your values are not continious (they are discrete since they can only be either a 1 'a 3 image' or a 0 'a 7 image')
def mse(predictions, targets): return ((predictions-targets)**2).mean()

def mnist_loss(predictions, targets): 
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

loss = mnist_loss(predictions, labels_test.float())

loss.backward()
weights.grad.shape,weights.grad.mean(),bias.grad

dl = DataLoader(dset_test, batch_size = 256)
#xb,yb = first(dl)
#xb.shape,yb.shape

dl_valid = DataLoader(dset_valid, batch_size = 256)

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(xb, yb)
    loss.backward()

def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()

def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in dl_valid]
    return round(torch.stack(accs).mean().item(), 4)

validate_epoch(linear1)

lr = 1
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')

# Creating an optimizer 

#nn.Linear does the same thing as our init_params and linear1() together. It contains both the weights and biases in a single class. Here's how we replicate our model from the previous section:

linear_model = nn.Linear(28*28,1)

w,b = linear_model.parameters()
w.shape,b.shape

class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None

opt = BasicOptim(linear_model.parameters(), lr)

def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()

validate_epoch(linear_model)

def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')

train_model(linear_model, 20)

# fastai provides the SGD class which, by default, does the same thing as our BasicOptim

linear_model = nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)

# fastai also provides Learner.fit, which we can use instead of train_model. To create a Learner we first need to create a DataLoaders, by passing in our training and validation DataLoaders:

dls = DataLoaders(dl, dl_valid)

# To create a Learner without using an application (such as vision_learner) we need to pass in all the elements that we've created in this chapter: the DataLoaders, the model, the optimization function (which will be passed the parameters), the loss function, and optionally any metrics to print:

learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

learn.fit(10, lr=lr)


# Adding a Nonlinearity
# A linear classifier is very constrained in terms of what it can do. To make it a bit more complex (and able to handle more tasks), we need to add something nonlinear between two linear classifiers—this is what gives us a neural network.
w1 = init_params((28*28,30))
b1 = init_params(30)
w2 = init_params((30,1))
b2 = init_params(1)

def simple_net(xb): 
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res

# The key point about this is that w1 has 30 output activations (which means that w2 must have 30 input activations, so they match). That means that the first layer can construct 30 different features, each representing some different mix of pixels. You can change that 30 to anything you like, to make the model more or less complex.

# That little function res.max(tensor(0.0)) is called a rectified linear unit, also known as ReLU. We think we can all agree that rectified linear unit sounds pretty fancy and complicated... But actually, there's nothing more to it than res.max(tensor(0.0))—in other words, replace every negative number with a zero. This tiny function is also available in PyTorch as F.relu:

# Just like in the previous section, we can replace this code with something a bit simpler, by taking advantage of PyTorch:

simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)

# nn.Sequential creates a module that will call each of the listed layers or functions in turn.

# nn.ReLU is a PyTorch module that does exactly the same thing as the F.relu function. Most functions that can appear in a model also have identical forms that are modules. Generally, it's just a case of replacing F with nn and changing the capitalization. When using nn.Sequential, PyTorch requires us to use the module version. Since modules are classes, we have to instantiate them, which is why you see nn.ReLU() in this example.

# Because nn.Sequential is a module, we can get its parameters, which will return a list of all the parameters of all the modules it contains. Let's try it out! As this is a deeper model, we'll use a lower learning rate and a few more epochs.

learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

learn.fit(40, 0.1)

# We're not showing the 40 lines of output here to save room; the training process is recorded in learn.recorder, with the table of output stored in the values attribute, so we can plot the accuracy over training as:

plt.plot(L(learn.recorder.values).itemgot(2))

learn.recorder.values[-1][2]

# There is no need to stop at just two linear layers. We can add as many as we want, as long as we add a nonlinearity between each pair of linear layers. As you will learn, however, the deeper the model gets, the harder it is to optimize the parameters in practice.

# With a deeper model (that is, one with more layers) we do not need to use as many parameters; it turns out that we can use smaller matrices with more layers, and get better results than we would get with larger matrices, and few layers.

# That means that we can train the model more quickly, and it will take up less memory.

dls = ImageDataLoaders.from_folder(path)
learn = vision_learner(dls, resnet18, pretrained=False,
                    loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)