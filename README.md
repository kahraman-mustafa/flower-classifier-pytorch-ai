# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch in Jupyter Notebook, then convert it into a command line application. In the project files 'flowers' image dataset folder is provided initially, image files in the dataset are used in training and testing neural network model to classify flower image to proper one of among 102 categories. 'cat_to_name.json' file is used to map numericall labels to name of categories.

This project is developed and run in the local machine. Conda environment with python 3.8 is used as a developmnet platform. All package and environmental dependencies are exported to `environment.yaml` file in root directory.

## Part 1:

Files required to run jupyter notebook and getting result for this part is only `Image Classifier Project.ipynb` notebook file. No local module is imported nor used in the notebook.

After running training, validation and testing cycles, trained model is saved as a checkpoint to load and use later. Saving directory is set project root directory as default in the notebook.

### Notes on Default Values

At the root directory there is folder named 'seek_bestparams'. This folder and the files in it are not required to run notebook. There are two python modules in the folder. For a list of hyperparameters, all combinations are trained sequentially by using inner for loops using these two python moules, namely `seek_bestparams_app.py` and `seek_bestparams_train.py`. Train and evalutaion accuracy and time results along with hyperparameter values are stored in `results.csv`. Accuracy values vs parameter combinations are plotted in `results.xlsx` and exported in the same folder to an image namely `accuaracy_results_mod.jpeg`. Parameters giving the best accuracy values for each model types is clearly seen that chart.

By analyzing this work results, defaults options are determined for this project. These parameter settings gave validation scores between 80%-90%.

- Default model: densenet161
  |** Default lr: 0.004
  |** Default n_epochs = 3
  |\_\_ Default n_hidden = 4096

- Other model: vgg16_bn
  |** Default lr: 0.002
  |** Default n_epochs = 3
  |\_\_ Default n_hidden = 4096

## Part 2

In this part, command line application is developed via two python modules; `train.py` and `predicty.py`. Files required to run console application and getting result for this part is only these files, and no other python moduls or etc is necessary to run these scripts neither.

`train.py` is used to train a model by allowing user' to choose model and set hyperparameters via command line arguments. It runs train, validation and test cycles behind the scenes and write to the console about loss, accuracy, epoch, timing, etc. status. Then it saves the model to default or user specified path. Default directory is _'./pth'_

`predict.py` is used to predict the classification desired number of highest likelihoods of provided image path. Image path, actual classification and predictions are written to console.

These two application is run via command line, for example:
`python train.py`
This line run train application with all default values

`python train.py --arch densenet161 --lr 0.002`
This line run train application with model architecture of densenet161 and learning rate of 0.002.

All command line arguments for these files are listed below:

`train.py` Command Line Arguments:

- Image Folder as `--imgdir` with default value _"flowers/"_
- Checkpoint saving folder as `--save_dir` with default value _"./"_
- Model Architecture as `--arch` with default value _"densenet161"_ choices of _[densenet161, vgg16_bn]_
- Number of hidden units as `--n_hidden` with default _4096_
- Learning rate as `--lr` with default value _0.004_
- Number of training epochs as `--n_epochs` with default value _3_
- Whether to use GPU for training as `--use_gpu` with default value _True_

`predict.py` Command Line Arguments:

- Image path as `--img_path` with default value of _randomly chosen image path_
- Checkpoint loading folder as `--load_dir` with default value _""_
- Whether to use GPU for training as `--use_gpu` with default value _True_
- Json File with flower category names as `--cat_names_json` with default value _"cat_to_name.json"_
- Number of classes having highest probabilities to display as `--topk` with default value of _5_
