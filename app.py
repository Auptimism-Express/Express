from werkzeug.utils import secure_filename
import base64
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask,flash,render_template, redirect, request, url_for
import numpy as np
from collections import OrderedDict,Counter
import numpy as np
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import models
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


plt.switch_backend('Agg')
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
app = Flask('Test')





def get_img(img_path):
    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    return img

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colours(img_path, no_of_colours, show_chart):
    img = get_img(img_path)
    #Reduce image size to reduce the execution time
    mod_img = cv2.resize(img, (600, 400), interpolation = cv2.INTER_AREA)
    #Reduce the input to two dimensions for KMeans
    mod_img = mod_img.reshape(mod_img.shape[0]*mod_img.shape[1], 3)

    #Define the clusters
    clf = KMeans(n_clusters = no_of_colours)
    labels = clf.fit_predict(mod_img)

    counts = Counter(labels)
    counts = dict(sorted(counts.items()))

    center_colours = clf.cluster_centers_
    ordered_colours = [center_colours[i] for i in counts.keys()]
    hex_colours = [RGB2HEX(ordered_colours[i]) for i in counts.keys()]
    rgb_colours = [ordered_colours[i] for i in counts.keys()]

    if (show_chart):
        fig=plt.figure(figsize = (6, 4))
        plt.pie(counts.values(), labels = hex_colours, colors = hex_colours)
        print("------------------")

        print(fig)
        print("------------------")

        plt.show()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")

        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return data
    else:
        return rgb_colours





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

    imgA = npImage[:, :, 0]
    imgB = npImage[:, :, 1]
    imgC = npImage[:, :, 2]

    imgA = (imgA - 0.485)/(0.229)
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)

    npImage[:, :, 0] = imgA
    npImage[:, :, 1] = imgB
    npImage[:, :, 2] = imgC

    npImage = np.transpose(npImage, (2, 0, 1))

    return npImage


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
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


def predict(img, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    loaded_model, class_to_idx = load_checkpoint('./emotions80_checkpoint.pth')
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # Implement the code to predict the class from an image file

    # image = torch.FloatTensor([process_image(Image.open(image_path))]) 

    image = torch.FloatTensor([process_image(Image.open(img))])
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


def view_classify(img):
    # class_names=['Happiness', 'Anxiety and Depression', 'Anger and Violence']
    # img = './static/img.jpg'
    loaded_model, class_to_idx = load_checkpoint('./emotions80_checkpoint.pth')
    # get_colours(img, 5, True)

    probabilities, classes = predict(img, loaded_model)

    ''' Function for viewing an image and it's predicted classes.
    '''
    img_filename = 'Prediction'
    img = Image.open(img)

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 6),  ncols=1, nrows=2)
    print("++++++++++")
    print(fig)
    print("++++++++++")

    ct_name = img_filename

    ax1.set_title(ct_name)
    ax1.imshow(img)
    ax1.axis('off')

    y_pos = np.arange(len(probabilities))
    ax2.barh(y_pos, probabilities, color='blue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(x for x in classes)
    ax2.invert_yaxis()

    # Save it to a temporary buffer.
    buf = io.BytesIO()
    fig.savefig(buf, format="png")

    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data

def plot_png():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = np.random.rand(100)
    ys = np.random.rand(100)
    axis.plot(xs, ys)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)


@app.route('/', methods=["GET", "POST"])
@app.route('/index', methods=["GET", "POST"])
def index():
    # return render_template('index.html', text='hello Shreyas', img_src=f"data:image/png;base64,{view_classify()}")
    return render_template('index.html')


app.config["UPLOAD_FOLDER"] = "static/Images"
#app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG","JFIF"]

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename="/Images" + filename), code=301)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_IMAGE_EXTENSIONS"]

@app.route('/home', methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":

        image = request.files['file']

        if image.filename == '':
            print("Image must have a file name")
            return redirect(request.url)


        filename = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        image.save(os.path.join(basedir,app.config["UPLOAD_FOLDER"],filename))
        img_path=os.path.join(basedir,app.config["UPLOAD_FOLDER"],filename)



    return render_template('output.html', text='hello Shreyas', img_src=f"data:image/png;base64,{view_classify(image)}",pie_chart=f"data:image/png;base64,{get_colours(img_path, 5, True)}")



app.run(debug=True,port=2000)