from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import copy
import argparse

def get_initial_data(data_dir):
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    }
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'valid', 'test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=8, shuffle=True, num_workers=4)
        for x in ['train', 'valid', 'test']
    }
    return image_datasets, dataset_sizes, dataloaders 

def get_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 7 command line arguments. If 
    the user fails to provide some or all of the 7 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
    1. Image Folder as --imgdir with default value "flowers/"
    2. Checkpoint saving folder as --save_dir with default value "./"
    3. Model Architecture as --arch with default value "densenet161" choices of [densenet161, vgg16_bn]
    4. Number of hidden units as --n_hidden with default 4096
    5. Learning rate as --lr with default value 0.004
    6. Number of training epochs as --n_epochs with default value 3
    7. Whether to use GPU for training as --use_gpu with default value True
    This function returns these arguments as an ArgumentParser object.
    Parameters:
    None - simply using argparse module to create & store command line arguments
    Returns:
    parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(
                    prog='Model Training Args',
                    description='Process model training arguments from user',
                    add_help=True)
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--img_dir',
                        type=str,
                        default='flowers/', 
                        help='Image Folder as --imgdir with default value "flowers/"')
    
    parser.add_argument('--save_dir',
                        type=str,
                        default='pth/', 
                        help='Checkpoint saving folder as --save_dir with default value "pth/"')
    
    parser.add_argument('--arch',
                        type=str,
                        default='densenet161',
                        choices=['densenet161', 'vgg16_bn'],
                        help='Model Architecture as --arch with default value "densenet" and choices of [densenet161, vgg16_bn]')
    
    parser.add_argument('--n_hidden',
                        type=int,
                        default=4096,
                        help='Number of hidden units with default 4096')
    
    parser.add_argument('--lr',
                        type=float,
                        default=0.004,
                        help='Learning rate with default value 0.004')
    
    parser.add_argument('--n_epochs',
                        type=int,
                        default=3,
                        help='Number of training epochs with default value 3')
    
    parser.add_argument('--use_gpu',
                        default=True,
                        action='store_true',
                        help='Whether to use GPU for training with default value True')

    return parser.parse_args()

def eval_model(dataloaders, dataset_sizes, device, model, criterion):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    
    test_batches = len(dataloaders['test'])
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in enumerate(dataloaders['test']):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        model.train(False)
        model.eval()
        inputs, labels = data

        with torch.no_grad():
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss_test += loss.item()
            acc_test += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
    avg_loss = loss_test / dataset_sizes['test']
    avg_acc = acc_test / dataset_sizes['test']
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)

def train_model(dataloaders, image_datasets, device, model, criterion, optimizer, n_epochs=3):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(dataloaders['train'])
    val_batches = len(dataloaders['valid'])
    
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch+1, n_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train(True)
        
        for i, data in enumerate(dataloaders['train']):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)
                
            inputs, labels = data
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        print()
        avg_loss = loss_train * 1 / len(image_datasets['train'])
        avg_acc = acc_train * 1 / len(image_datasets['train'])
        
        model.train(False)
        model.eval()
            
        for i, data in enumerate(dataloaders['valid']):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                
            inputs, labels = data
            
            with torch.no_grad():
                inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                loss_val += loss.item()
                acc_val += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / len(image_datasets['valid'])
        avg_acc_val = acc_val / len(image_datasets['valid'])
        
        print()
        print("Epoch {} result: ".format(epoch+1))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()
        # Check if the avg accuracy at the end of current epoch is higher than 
        # the best accuracy until that epoch, if so, set it as new best accuracy value
        # and make a copy of the state dict of the model 
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model

def set_classifier(device, n_categories, arch, n_hidden, lr):
    DENSENET161 = 'densenet161'
    VGG16BN = 'vgg16_bn'

    densenet161_trained_wieghts = models.DenseNet161_Weights.DEFAULT
    vgg16bn_trained_weights = models.VGG16_BN_Weights.DEFAULT

    densenet161_model = models.densenet161 
    vgg16bn_model = models.vgg16_bn
    
    # Rubric: A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
    if arch == DENSENET161:
        model_train = densenet161_model(densenet161_trained_wieghts)
    elif arch == VGG16BN:
        model_train = vgg16bn_model(vgg16bn_trained_weights)

    # Freeze parameters so we don't backprop through them
    for param in model_train.parameters():
        param.requires_grad = False
    
    if arch == DENSENET161:
        n_features = 2208
        model_train.classifier = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_categories), # Output layer for 102 flower categories
            nn.LogSoftmax(dim=1))
    elif arch == VGG16BN:
        n_features = 4096
        features = list(model_train.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(n_features, n_hidden),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=0.2, inplace=False),
                            nn.Linear(n_hidden, n_categories), # Output layer for 102 flower categories
                            nn.LogSoftmax(dim=1)]) # Output layer with 102 outputs
        model_train.classifier = nn.Sequential(*features) # Replace the model classifier
    
    model_train.to(device)

    criterion = nn.NLLLoss()
    
    # Set optimizer and learning rate scheduler
    optimizer_ft = optim.SGD(model_train.parameters(), lr=lr, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return model_train, criterion, optimizer_ft

def save_checkpoint(image_datasets, device, save_dir, arch, model, n_hidden, optimizer, lr, n_epochs):
    ''' Save the trained and tested network '''
    model.cpu()
    checkpoint_path = f'{arch}_flowerclassifier.pth'
    checkpoint = {'arch': arch,
                'n_hidden': n_hidden,
                'class_to_idx': image_datasets['train'].class_to_idx,
                'optimizer_dict': optimizer.state_dict(),
                'classifier': model.classifier,
                'lr': lr,
                'n_epochs': n_epochs,
                'state_dict': model.state_dict()}
    torch.save(checkpoint, os.path.join(save_dir, checkpoint_path))
    model.to(device)

def main():
    ''' Create & retrieve Command Line arguments. '''
    in_args = get_input_args()
    arch = in_args.arch
    n_hidden = in_args.n_hidden
    lr = in_args.lr
    n_epochs = in_args.n_epochs
    data_dir = in_args.img_dir
    device = torch.device("cuda" if (torch.cuda.is_available() and in_args.use_gpu) else "cpu") # Use GPU if it's available
    save_dir = in_args.save_dir
    
    image_datasets, dataset_sizes, dataloaders = get_initial_data(data_dir)
    class_names = image_datasets['train'].classes
    n_categories = len(class_names)

    print(f"\nmodel:{arch}, n_hidden:{n_hidden}, lr:{lr}, n_epochs: {n_epochs}")
    model_to_train, criterion, optimizer_ft = set_classifier(device, n_categories, arch, n_hidden, lr)
    model_trained = train_model(dataloaders, image_datasets, device, model_to_train, criterion, optimizer_ft, n_epochs)
    eval_model(dataloaders, dataset_sizes, device, model_trained, criterion)
    save_checkpoint(image_datasets, device, save_dir, arch, model_trained, n_hidden, optimizer_ft, lr, n_epochs)

def run():
    torch.multiprocessing.freeze_support()
    print('loop')
    
if __name__ == '__main__':
    # run()
    main()