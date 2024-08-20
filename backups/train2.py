from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import copy
import json
from helper import InputTaker

class Train():
    def __init__(self):
        data_dir = 'flowers'
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'
        self.data_transforms = {
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
        self.image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(data_dir, x), self.data_transforms[x])
            for x in ['train', 'valid', 'test']
        }
        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.image_datasets[x], batch_size=8, shuffle=True, num_workers=4)
            for x in ['train', 'valid', 'test']
        }
        self.class_names = self.image_datasets['train'].classes
        self.n_categories = len(self.class_names)
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f) # Category number (as string) to category name
        self.class_labels = [cat_to_name[c] for c in self.class_names]
        self.device = 'cpu'

    def train_model(self, model, criterion, optimizer, scheduler, n_epochs=5):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        avg_loss = 0
        avg_acc = 0
        avg_loss_val = 0
        avg_acc_val = 0
        
        train_batches = len(self.dataloaders['train'])
        val_batches = len(self.dataloaders['valid'])
        
        for epoch in range(n_epochs):
            print("Epoch {}/{}".format(epoch+1, n_epochs))
            print('-' * 10)
            
            loss_train = 0
            loss_val = 0
            acc_train = 0
            acc_val = 0
            
            model.train(True)
            
            for i, data in enumerate(self.dataloaders['train']):
                if i % 100 == 0:
                    print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)
                    
                # # Use half training dataset
                # if i >= train_batches / 2:
                #     break
                    
                inputs, labels = data
                inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))
                
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
            # * 2 as we only used half of the dataset
            avg_loss = loss_train * 1 / len(self.image_datasets['train'])
            avg_acc = acc_train * 1 / len(self.image_datasets['train'])
            
            model.train(False)
            model.eval()
                
            for i, data in enumerate(self.dataloaders['valid']):
                if i % 100 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                    
                inputs, labels = data
                
                with torch.no_grad():
                    inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))
                    
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    
                    loss_val += loss.item()
                    acc_val += torch.sum(preds == labels.data)
                    
                    del inputs, labels, outputs, preds
                    torch.cuda.empty_cache()
            
            avg_loss_val = loss_val / len(self.image_datasets['valid'])
            avg_acc_val = acc_val / len(self.image_datasets['valid'])
            
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
    
    def main(self):
        print(f'Number of categories: {self.n_categories}')
        print(f"Classes: {self.class_names}")
        print(f"Classes as text: {self.class_labels}")
        
        # TODO: Build and train your network
        # Use GPU if it's available and according to user preference
        inp_dp = InputTaker.get_gpu_ref()
        device_prefs = {1: 'cpu', 2: 'gpu'}
        device_pref = device_prefs[inp_dp]
        self.device = torch.device("cuda" if (torch.cuda.is_available() and device_pref == 'gpu') else "cpu")
        print(f'Device preference: {self.device}\n')
        
        DENSENET161 = 'densenet161'
        VGG16BN = 'vgg16_bn'

        densenet161_trained_wieghts = models.DenseNet161_Weights.DEFAULT
        vgg16bn_trained_weights = models.VGG16_BN_Weights.DEFAULT

        densenet161_model = models.densenet161 
        vgg16bn_model = models.vgg16_bn

        inp_mp = InputTaker.get_model_pref()
        model_prefs = {1: DENSENET161, 2: VGG16BN}
        model_pref = model_prefs[inp_mp]
        print(f'Model preference: {model_pref}\n')
        
        # Rubric: A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
        if model_pref == DENSENET161:
            model_train = densenet161_model(densenet161_trained_wieghts)
        elif model_pref == VGG16BN:
            model_train = vgg16bn_model(vgg16bn_trained_weights)

        # Freeze parameters so we don't backprop through them
        for param in model_train.parameters():
            param.requires_grad = False
            
        print(f'Transferred model: {model_train}\n')
        
        inp_n_hidden = InputTaker.get_n_hidden_pref()
        if model_pref == DENSENET161:
            n_features = 2208
            model_train.classifier = nn.Sequential(
                nn.Linear(n_features, inp_n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(inp_n_hidden, self.n_categories), # Output layer for 102 flower categories
                nn.LogSoftmax(dim=1))
        elif model_pref == VGG16BN:
            n_features = 4096
            features = list(model_train.classifier.children())[:-1] # Remove last layer
            features.extend([nn.Linear(n_features, inp_n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.2, inplace=False),
                                nn.Linear(inp_n_hidden, self.n_categories), # Output layer for 102 flower categories
                                nn.LogSoftmax(dim=1)]) # Output layer with 102 outputs
            model_train.classifier = nn.Sequential(*features) # Replace the model classifier
            
        print(f'Modified classifier: {model_train.classifier}\n')
        
        model_train.to(self.device)

        criterion = nn.NLLLoss()
        
        inp_lr = InputTaker.get_learnrate_pref()
        print(f'Learn rate preference: {inp_lr}\n')
        
        inp_mm = InputTaker.get_momentum_pref()
        print(f'Momentum preference: {inp_mm}\n')
        
        inp_ne = InputTaker.get_n_epochs()
        print(f'Epochs numer preference: {inp_ne}\n')
        
        # Set optimizer and learning rate scheduler
        optimizer_ft = optim.SGD(model_train.parameters(), lr=inp_lr, momentum=inp_mm)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        
        # Train model
        print(f'''
            > Model = {model_pref}
            > Hidden units: {inp_n_hidden}
            > Learning rate: {inp_lr}
            > Epochs: {inp_ne}
            > Momentum: {inp_mm}
            > Batchsize: {8}
            ''')
        model_trained = self.train_model(model_train, criterion, optimizer_ft, exp_lr_scheduler, n_epochs=inp_ne)
        
    def run():
        torch.multiprocessing.freeze_support()
        print('loop')
        
    if __name__ == '__main__':
        run()
        main()