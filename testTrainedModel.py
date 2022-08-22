from torchvision import models
from PIL import Image
import argparse
import torch.nn as  nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torchvision.transforms as standard_transforms

labelNames = ['3D objects',
    'Algorithm',
    'Area chart',
    'Bar plots',
    'Block diagram',
    'Box plot',
    'Bubble Chart',
    'Confusion matrix',
    'Contour plot',
    'Flow chart',
    'Geographic map',
    'Graph plots',
    'Heat map',
    'Histogram',
    'Mask',
    'Medical images',
    'Natural images',
    'Pareto charts',
    'Pie chart',
    'Polar plot',
    'Radar chart',
    'Scatter plot',
    'Sketches',
    'Surface plot',
    'Tables',
    'Tree Diagram',
    'Vector plot',
    'Venn Diagram']



def get_parser():
    parser = argparse.ArgumentParser(description='DocFigure trained model')
    parser.add_argument('-f','--trainedFigClassModel', type=str,
                 default='/path/to/epoch_9_loss_0.04706_testAcc_0.96867_X_resnext101_docSeg.pth')
    parser.add_argument('-i','--inputImage', type=str, default='basic-bar-graph.png')
    return parser

def fig_classification(fig_class_model_path):
    fig_model =  models.resnext101_32x8d()
    num_features = fig_model.fc.in_features
    fc = list(fig_model.fc.children()) # Remove last layer
    fc.extend([nn.Linear(num_features, 28)]) # Add our layer with 4 outputs
    fig_model.fc = nn.Sequential(*fc)
    fig_model = fig_model.to(device)
    fig_model.load_state_dict(torch.load(fig_class_model_path))
    fig_model.eval()
    mean_std = ( [.485, .456, .406], [.229, .224, .225])
    fig_class_trasform = standard_transforms.Compose([
        standard_transforms.Resize((384, 384), interpolation=Image.ANTIALIAS),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)         ])
    return fig_model, fig_class_trasform



if __name__ == "__main__":
    args = get_parser().parse_args()
    img_path    = args.inputImage
    image = Image.open(img_path).convert('RGB')
    # figure classification model
    fig_model, fig_class_trasform = fig_classification(args.trainedFigClassModel)
    img_tensor = fig_class_trasform(image)
    fig_label = fig_model(img_tensor.cuda().unsqueeze(0))
    fig_prediction = fig_label.max(1)[1]
    out_put =labelNames[fig_prediction]
    print ('The detected document class is ',out_put)
