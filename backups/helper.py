import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import os


def test_network(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')

def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

class InputTaker():

    def get_gpu_ref():
        while True: 
            try: 
                inp_dp = int(input("\nChoose and type GPU usage option (-1 for default):\n1: Use CPU \n2: Use GPU if available\n"))
                if inp_dp == -1: # Default is use gpu if available
                    inp_dp = 2
                    break
                if inp_dp in [1, 2]: 
                    break 
                else: 
                    print("Either 1 or 2 must be entered.") 
            except ValueError: 
                print("Invalid input. Please enter an integer.") 
        return inp_dp

    def get_model_pref():
        while True: 
            try: 
                inp_mp = int(input("Choose pretrained model (-1 for default):\n1: densetnet161 \n2: vgg16_bn\n")) 
                if inp_mp == -1: # Default is vgg16_bn
                    inp_mp = 2
                    break
                if inp_mp in [1, 2]: 
                    break 
                else: 
                    print("Either 1 or 2 must be entered.") 
            except ValueError: 
                print("Invalid input. Please enter an integer.")
        return inp_mp

    def get_n_hidden_pref():
        while True: 
            try: 
                inp_n_hidden = int(input("Enter number of hidden units (value range [102, 4096]) (-1 for default):\n")) 
                if inp_n_hidden == -1: # Default is 1024
                    inp_n_hidden = 1024
                    break
                if inp_n_hidden >= 102 and inp_n_hidden <= 4096: # inputs are limited to this range
                    break 
                else: 
                    print("Number of hidden units should be in value range [102, 4096]") 
            except ValueError: 
                print("Invalid input. Please enter an integer.")
        return inp_n_hidden

    def get_learnrate_pref():
        # Allow user to set learning rate
        while True: 
            try: 
                inp_lr = float(input("Enter learning rate (value range [0.001, 0.1]) (-1 for default):\n")) 
                if inp_lr == -1: # Default is 0.002
                    inp_lr = 0.002
                    break
                if inp_lr >= 0.001 and inp_lr <= 0.1: # inputs are limited to this range
                    break 
                else: 
                    print("Learning rate should be in value range [0.001, 0.1]") 
            except ValueError: 
                print("Invalid input. Please enter an integer.")
        return inp_lr
                
    def get_momentum_pref():
        # Allow user to set momentum factor
        while True: 
            try: 
                inp_mm = float(input("Enter momentum factor (value range (0, 1)) (-1 for default):\n"))
                if inp_mm == -1: # Default is 0.9
                    inp_mm = 0.9
                    break
                if inp_mm > 0 and inp_mm < 1: # inputs are limited to this range
                    break 
                else: 
                    print("Momentum factor should be in value range (0, 1)") 
            except ValueError: 
                print("Invalid input. Please enter an integer.")
        return inp_mm

    def get_n_epochs():
        # Allow user to set number of epochs
        while True: 
            try:
                inp_ne = int(input("Enter number of epochs (value range [1, 5]) (-1 for default):\n"))
                if inp_ne == -1: # Default is 3
                    inp_ne = 3
                    break
                if inp_ne >= 1 and inp_ne <= 5: # inputs are limited to this range
                    break 
                else: 
                    print("Number of epochs should be in value range [1, 5]") 
            except ValueError: 
                print("Invalid input. Please enter an integer.")
        return inp_ne
    
    def get_cat_json_path():
        return 'cat_to_name.json'
    
    def get_pth_file():
        # To display in input dialog find list of saved pth files in root directory
        pth_files = [file for file in os.listdir(path='./') if file.endswith('.pth')]
        # print(pth_files)
        n_checkpoint = len(pth_files)
        
        # Create input dialog options
        dialog_text = ""
        for i in range(n_checkpoint):
            dialog_text += f"\n{i+1}: {pth_files[i]}"
            
        # Allow user to set number of epochs
        while True: 
            try:
                inp_check = int(input(f"\nChoose trained model (-1 for default): {dialog_text}"))
                if inp_check == -1: # Default is 1
                    inp_check = 1
                    break
                if inp_check in range(1, n_checkpoint+1): # inputs are limited to this range
                    break 
                else: 
                    print("Invalid model choice") 
            except ValueError: 
                print("Invalid input. Please enter an integer.")
        return inp_check