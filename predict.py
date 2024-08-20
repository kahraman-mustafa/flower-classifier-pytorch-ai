import argparse
import torch
from torchvision import models
import numpy as np
import json
import os
from PIL import Image

# To test process_image function via randomly generated image paths
def random_img_path(imgset='test'):
    data_dir = 'flowers'
    path = os.path.join(data_dir, imgset)
    list_dir_class = os.listdir(path)
    rand_class = list_dir_class[np.random.randint(0, len(list_dir_class))]
    path_rand_class = os.path.join(path, rand_class)
    list_dir_img = os.listdir(path_rand_class)
    rand_img = list_dir_img[np.random.randint(0, len(list_dir_img))]
    path_rand_img = os.path.join(path_rand_class, rand_img)
    return path_rand_img

def get_pth_file(path):
    pth_files = [file for file in os.listdir(path=path) if file.endswith('.pth')]
    if len(pth_files) > 0:
        return os.path.join(path, pth_files[0])
    else:
        return ''

def get_input_args():
    """
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 5 command line arguments. If 
    the user fails to provide some or all of the 5 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
    1. Image path as --img_path with default value of randomly chosen image path
    2. Checkpoint loading folder as --load_dir with default value "./"
    3. Whether to use GPU for training with default value True
    4. Json File with flower category names as --cat_names_json with default value "cat_to_name.json"
    5. Number of classes having highest probabilities to display as --topk with default value of 5
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
    parser.add_argument('--img_path',
                        type=str,
                        default=random_img_path(), 
                        help='Image path as --img_path with default value of randomly chosen image path')
    
    parser.add_argument('--load_path',
                        type=str,
                        default=get_pth_file('pth'), 
                        help='Checkpoint loading path as --load_path with default value "pth"')
    
    parser.add_argument('--use_gpu',
                        default=True,
                        action='store_true',
                        help='Whether to use GPU for training with default value True')
    
    parser.add_argument('--cat_names_json', 
                        type=str,
                        default='cat_to_name.json', 
                        help='Json File with flower category names as --cat_names_json with default value "cat_to_name.json"')
    
    parser.add_argument('--topk', 
                        type=int,
                        default=5, 
                        help='Number of classes having highest probabilities to display as --topk with default value of 5')

    return parser.parse_args()

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path, device):
    VGG16BN = 'vgg16_bn'
    DENSENET161 = 'densenet161'
    try:
        checkpoint = torch.load(path)
    except:
        print("Invalid checkpoint path.")
        return None
    
    if checkpoint['arch'] == VGG16BN:
        model = models.vgg16_bn()
    elif checkpoint['arch'] == DENSENET161:
        model = models.densenet161()
    else:
        print("Sorry base architecture note recognized")
        return None
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model

def process_image(image):
    ''' Scale, crop, and normalize a PIL image for a PyTorch model,
        and return the data in an Numpy array. '''
    w, h = image.size
    # Resize
    if w > h:
        w_ = int(w*(256/h))
        image.resize((w_, 256))
    else:
        h_ = int(h*(256/w))
        image.resize((256, h_))
    # Crop 
    left_margin = (image.width-224)/2
    bottom_margin = (image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    image = image.crop((left_margin, bottom_margin, right_margin, 
                    top_margin))
    # Normalize
    image = np.array(image)
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    image = (image/255 - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose((2, 0, 1))
    return image


def predict(device, image_path, model, cat_to_name, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    model.eval()
    print(f'\nImage path: {image_path}')
    image = Image.open(image_path)
    image = torch.tensor(process_image(image))
    image.unsqueeze_(0)
    model = model.double()
    
    # Convert class_to_idx keys and values to actual numerical classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    model.to(device)
    with torch.no_grad():
        output = model(image.to(device))
        top_ps, top_cats = torch.exp(output).data.topk(topk)
    probs = top_ps.tolist()[0]
    categories = list(map(lambda i:idx_to_class[i], top_cats.tolist()[0]))
    cat_names = [cat_to_name[cat] for cat in categories]
    
    return probs, categories, cat_names

def main():
    ''' Create & retrieve Command Line arguments. '''
    in_args = get_input_args()
    topk = in_args.topk
    img_path = in_args.img_path
    json_path = in_args.cat_names_json
    device = torch.device("cuda" if (torch.cuda.is_available() and in_args.use_gpu) else "cpu") # Use GPU if it's available
    load_path = in_args.load_path
    
    img_class = img_path.split('\\')[2] # Extract actual class from image path 
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f) # Category number (as string) to category name
    img_class_name = cat_to_name[img_class]
    
    model_load = load_checkpoint(load_path, device)
    if model_load is None:
        return None
    probs, classes, cat_names = predict(device, img_path, model_load, cat_to_name, topk)
    print(f'Actual Class -> {img_class}: {img_class_name}')
    print(f'Prediction ->')
    class_n_name = []
    for i in range(len(classes)):
        combined = classes[i] + ':' + cat_names[i]
        class_n_name.append(f'{combined:_<30}')
    print(json.dumps(dict(zip(class_n_name, np.round(probs, 3))), indent=1))

if __name__ == '__main__':
	main()