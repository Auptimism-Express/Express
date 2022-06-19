from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model

    size = 361, 361
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.
        
    imgA = npImage[:,:,0]
    imgB = npImage[:,:,1]
    imgC = npImage[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
        
    npImage[:,:,0] = imgA
    npImage[:,:,1] = imgB
    npImage[:,:,2] = imgC
    
    npImage = np.transpose(npImage, (2,0,1))
    
    return npImage

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location='cpu')
    model = models.resnet152()
    
    # our input_size matches the in_features of pretrained model
    input_size = 2048
    output_size = 5
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 1024)),
                          ('relu', nn.ReLU()),
                          #('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(1024, 3)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


    # replacing the pretrained model classifier with our classifier
    model.fc = classifier
    
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']

loaded_model, class_to_idx = load_checkpoint('./emotions80_checkpoint.pth')
idx_to_class = { v : k for k,v in class_to_idx.items()}
class_names=['Happiness', 'Anxiety and Depression', 'Anger and Violence']

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    
    image = torch.FloatTensor([process_image(Image.open(image_path))])
    model.eval()
    output = model.forward(Variable(image))
    print(output)
    probabilities = torch.exp(output).data.numpy()[0]
    

    top_idx = np.argsort(probabilities)[-topk:][::-1] 
    print(top_idx)
    top_class = [idx_to_class[x] for x in top_idx]
    print(top_class)
    top_probability = probabilities[top_idx]

    return top_probability, top_class

def view_classify(img, probabilities, classes, mapper):

    ''' Function for viewing an image and it's predicted classes.
    '''
    img_filename = 'Prediction'
    img = Image.open(img)

    fig, (ax1, ax2) = plt.subplots(figsize=(15,10),  ncols=1, nrows=2)
    ct_name = img_filename
    
    ax1.set_title(ct_name)
    ax1.imshow(img)
    ax1.axis('off')

    y_pos = np.arange(len(probabilities))
    ax2.barh(y_pos, probabilities, color='blue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(x for x in classes)
    ax2.invert_yaxis()
    print("hi..................!!!!!!!!!")
    # plt.figure(figsize=(5,16))
    plt.show()

    

    # fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)
    # xs = np.random.rand(100)
    # ys = np.random.rand(100)
    # axis.plot(xs, ys)
    # output = io.BytesIO()
    # FigureCanvas(fig).print_png(output)
    # plt.show()
    
    # return Response(output.getvalue(), mimetype='image/png')


img = './Children_emotions/Valid/Anger and Violence/2.jpg'
p, c = predict(img, loaded_model)
view_classify(img, p, c, class_names)