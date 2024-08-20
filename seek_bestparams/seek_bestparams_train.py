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

class Train():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if it's available
        self.data_dir = 'flowers'
        self.train_dir = self.data_dir + '/train'
        self.valid_dir = self.data_dir + '/valid'
        self.test_dir = self.data_dir + '/test'
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
                os.path.join(self.data_dir, x), self.data_transforms[x])
            for x in ['train', 'valid', 'test']
        }
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'valid', 'test']}
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
        
    def eval_model(self, model, criterion):
        since = time.time()
        avg_loss = 0
        avg_acc = 0
        loss_test = 0
        acc_test = 0
        
        test_batches = len(self.dataloaders['test'])
        print("Evaluating model")
        print('-' * 10)
        
        for i, data in enumerate(self.dataloaders['test']):
            if i % 100 == 0:
                print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

            model.train(False)
            model.eval()
            inputs, labels = data

            with torch.no_grad():
                inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))

                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                loss_test += loss.item()
                acc_test += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            
        avg_loss = loss_test / self.dataset_sizes['test']
        avg_acc = acc_test / self.dataset_sizes['test']
        
        elapsed_time = time.time() - since
        print()
        print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        print("Avg loss (test): {:.4f}".format(avg_loss))
        print("Avg acc (test): {:.4f}".format(avg_acc))
        print('-' * 10)

    def train_model(self, model, criterion, optimizer, scheduler, n_epochs=3):
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
        return model, avg_acc.item(), best_acc.item(), elapsed_time
    
    def main(self, md, nh, lr, ne):
        DENSENET161 = 'densenet161'
        VGG16BN = 'vgg16_bn'

        densenet161_trained_wieghts = models.DenseNet161_Weights.DEFAULT
        vgg16bn_trained_weights = models.VGG16_BN_Weights.DEFAULT

        densenet161_model = models.densenet161 
        vgg16bn_model = models.vgg16_bn
        
        # Rubric: A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
        if md == DENSENET161:
            model_train = densenet161_model(densenet161_trained_wieghts)
        elif md == VGG16BN:
            model_train = vgg16bn_model(vgg16bn_trained_weights)

        # Freeze parameters so we don't backprop through them
        for param in model_train.parameters():
            param.requires_grad = False
        
        if md == DENSENET161:
            n_features = 2208
            model_train.classifier = nn.Sequential(
                nn.Linear(n_features, nh),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(nh, self.n_categories), # Output layer for 102 flower categories
                nn.LogSoftmax(dim=1))
        elif md == VGG16BN:
            n_features = 4096
            features = list(model_train.classifier.children())[:-1] # Remove last layer
            features.extend([nn.Linear(n_features, nh),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.2, inplace=False),
                                nn.Linear(nh, self.n_categories), # Output layer for 102 flower categories
                                nn.LogSoftmax(dim=1)]) # Output layer with 102 outputs
            model_train.classifier = nn.Sequential(*features) # Replace the model classifier
        
        model_train.to(self.device)

        criterion = nn.NLLLoss()
        
        # Set optimizer and learning rate scheduler
        optimizer_ft = optim.SGD(model_train.parameters(), lr=lr, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        # Train model
        model_trained, acc_train, acc_valid, elapsed_time = self.train_model(model_train, criterion, optimizer_ft, exp_lr_scheduler, n_epochs=ne)
        return acc_train, acc_valid, elapsed_time
        
    def run():
        torch.multiprocessing.freeze_support()
        print('loop')
        
    if __name__ == '__main__':
        # run()
        main()