
import os.path
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import time


def get_input_args():
    ''' Retrieve and parse the command line arguments created and defined using the argparse module. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to image folders')
    parser.add_argument('--save_dir', default='.', help='directory to checkpoint')
    parser.add_argument('--arch', default='densenet', type=str, choices=['densenet', 'vgg'], help='pre-trained model architecture: choice of densenet, vgg')
    parser.add_argument('--hidden_units', default=512, type=int, help='number of hidden units')
    parser.add_argument('--learning_rate', default=0.003, type=float, help='learning rate')
    parser.add_argument('--epochs', default=3, type=int, help='number of training epochs')
    parser.add_argument('--gpu', default=True, action='store_true', help='use GPU for training')
    
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'pet_images'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
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
                        default='./', 
                        help='Checkpoint saving folder as --save_dir with default value "./"')
    
    parser.add_argument('--arch',
                        type=str,
                        default='densenet',
                        choices=['densenet', 'vgg'],
                        help='Model Architecture as --arch with default value "densenet" and choices of [densenet, vgg]')
    
    parser.add_argument('--n_hidden',
                        type=int,
                        default=4096,
                        help='Number of hidden units')
    
    parser.add_argument('--lr',
                        type=float,
                        default=0.004,
                        help='Learning rate')
    
    parser.add_argument('--n_epochs',
                        type=int,
                        default=3,
                        help='Number of training epochs')
    
    parser.add_argument('--use_gpu',
                        default=True,
                        action='store_true',
                        help='Whether to use GPU for training')
    
    parser.add_argument('--catnames', 
                        type=str,
                        default='cat_to_name.json', 
                        help='Json File with flower category names as --catnames with default value "cat_to_name.json"')

    return parser.parse_args()


def dataset_loaders(data_dir):
    '''Load, split, transform and define the relevant datasets. '''
    # Separate the data into training, validation, and test sets
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)

    # Load the mapping of classes to indices from the training dataset
    class_to_idx = train_datasets.class_to_idx

    return trainloader, validloader, testloader, class_to_idx


def model_arch(arch):
    ''' Retrieve a pre-trained model architecture and match it with its respective input size. '''
    if arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'resnet':
        model = models.resnet18(pretrained=True)
        input_size = model.fc.in_features
    elif arch == 'vgg':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    else:
        raise ValueError('{} architecture is not supported by this program. Please choose from: densenet, resnet, vgg.'.format(arch))

    return model, input_size


def build_classifier(arch, model, input_size, hidden_units, learning_rate):
    ''' Build a classifier with the specified pre-trained model, input size, number of
    hidden units, and learning rate.'''

    # Freeze parameters so the program doesn't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Build a feed-forward classifier with ReLU, dropout and Softmax on the output
    output_size = 102
    classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_units, output_size),
                               nn.LogSoftmax(dim=1))
    
    # Allow for resnet's lack of classifier attribute to replace
    if arch == 'resnet':
        model.fc = classifier
    else:
        model.classifier = classifier

    # Define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer


def train_classifier(model, trainloader, validloader, criterion, optimizer, epochs, device, print_every=5):
    ''' Train the network with a cross-validation pass. '''
    steps = 0
    running_loss = 0

    model.to(device)

    for e in range(epochs):
        model.train()

        for images, labels in trainloader:
            steps += 1
            # Transfer training data to GPU functioning, when available
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()

            # Initiate the validation pass in the loop
            if steps % print_every == 0:
            # Make sure network is in eval mode for inference
                model.eval()

                valid_loss = 0
                accuracy = 0

                for images, labels in validloader:
                    # Transfer vailidation data to GPU functioning, when available
                    images, labels = images.to(device), labels.to(device)

                    # Turn off gradients for validation, to save memory and computations
                    with torch.no_grad():
                        output = model.forward(images)
                        valid_loss += criterion(output, labels).data.item()

                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                print('Epoch: {}/{}.. '.format(e+1, epochs),
                      'Training Loss: {:.3f}.. '.format(running_loss/print_every),
                      'Validation Loss: {:.3f}.. '.format(valid_loss/len(validloader)),
                      'Validation Accuracy: {:.3f}'.format(accuracy/len(validloader)))

                running_loss = 0
                # Make sure training is back on
                model.train()

    print('Training is complete!')
    return model


def test_classifier(model, testloader, criterion, device):
    ''' Implement the test pass for the trained model '''
    # Do validation on the test set
    model.eval()

    test_loss = 0
    accuracy = 0

    for images, labels in testloader:
        # Transfer test data to GPU functioning, when available
        images, labels = images.to(device), labels.to(device)

        # Turn off gradients for validation, to save memory and computations
        with torch.no_grad():
            output = model(images)
            test_loss += criterion(output, labels).data.item()

            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    print('Test Loss: {:.3f}.. '.format(test_loss/len(testloader)),
          'Test Accuracy: {:.3f}'.format(accuracy/len(testloader)))
    
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


def save_checkpoint(arch, model, input_size, output_size, hidden_units, class_to_idx, optimizer, learning_rate, epochs, save_dir):
    ''' Save the trained and tested network '''
    checkpoint = {'arch': arch,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_units': hidden_units,
                  'class_to_idx': class_to_idx,
                  'optimizer_dict': optimizer.state_dict(),
                  'classifier': model.classifier,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))


def main():
    ''' Create & retrieve Command Line arguments. '''
    in_args = get_input_args()
    output_size = 102
    # Enable GPU functionality, when available
    device = torch.device('cuda' if torch.cuda.is_available() and in_args.gpu else 'cpu')
    data_dir = in_args.data_dir
    model, input_size = model_arch(in_args.arch)
    model, criterion, optimizer = build_classifier(in_args.arch, model, input_size, in_args.hidden_units, in_args.learning_rate)
    trainloader, validloader, testloader, class_to_idx = dataset_loaders(data_dir)
    trained_model = train_classifier(model, trainloader, validloader, criterion, optimizer, in_args.epochs, device)
    test_classifier(model, testloader, criterion, device)
    save_checkpoint(in_args.arch, trained_model, input_size, output_size, in_args.hidden_units, class_to_idx, optimizer,
                    in_args.learning_rate, in_args.epochs, in_args.save_dir)

if __name__ == '__main__':
    main()