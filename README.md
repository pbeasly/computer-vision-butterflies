## Butterfly Classification and Segmentation
The project performs Deep Learning training on a set of butterfly images to extract a classification model and a segmentation model.  The dataset is downloaded from a Kaggle repository [Butterfly Dataset](https://www.kaggle.com/datasets/veeralakrishna/butterfly-dataset).  
### Original work citation:
Josiah Wang, Katja Markert, and Mark Everingham
Learning Models for Object Recognition from Natural Language Descriptions
In Proceedings of the 20th British Machine Vision Conference (BMVC2009)

## Training Approach
The training is performed using Python's Pytorch library.  The ___ notebook can be run in Google's Colab to relatively quickly train the models if using the GPU run session. Details can be found in the Jupyter Notebook.  Both Multilayer Perceptron (MLP) and CNN-RESNET (Convolutional Network with Residual Layers) were used to perform the training.  Validation accuracy 

The training uses the Pytorch library in Python and compares different models for accuracy in a validation set.  The ___ notebook runs the dataloader and training scripts to generate the results and models.

## Example images
![Butterfly Dataset Visualization](image_segmentation.png)

