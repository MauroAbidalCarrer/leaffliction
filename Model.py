from torch import nn
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torchvision.tv_tensors import Image
from torchvision.transforms import ToTensor

test = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(3969, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10), 
            nn.Softmax(1),
        )

img = read_image('/mnt/nfs/homes/maabidal/Downloads/astronaut.jpg')
img = resize(img, 128, antialias=True)
img = img.float()
print(test.forward(img))