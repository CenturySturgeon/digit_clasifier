#hide
# ! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastai.vision.all import *
from fastbook import *

matplotlib.rc('image', cmap='Greys')

path = untar_data(URLs.MNIST_SAMPLE)

Path.BASE_PATH = path

(path/'train').ls()

threes_tensors = [tensor(Image.open(o)) for o in (path/'train'/'3').ls().sorted()]
stacked_threes = torch.stack(threes_tensors).float()/255
#stacked_threes.shape

sevens_tensors = [tensor(Image.open(o)) for o in (path/'train'/'7').ls().sorted()]
stacked_sevens = torch.stack(sevens_tensors).float()/255
#stacked_sevens.shape

#reshape the tensors and concatenate them in a new one
test_tensor = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
test_tensor.shape

valid3s_tensor = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()]).float()/255
valid7s_tensor = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()]).float()/255
validation_tensor = torch.cat([valid3s_tensor, valid7s_tensor]).view(-1, 28*28)
#validation_tensor.shape