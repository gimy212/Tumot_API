from flask import Flask, request, render_template, url_for, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.utils import load_img, img_to_array


app = Flask(__name__)

# Plant_Diseases Classification Preprocessing
def img_preprossing(image):
    image=Image.open(image)
    image = image.resize((200, 200))
    image_arr = np.array(image.convert('L'))
    image_arr.shape = (1, 200, 200, 1)
    image_arr = image_arr / 255.0
    return image_arr

tumor_model = load_model('test.h5')

# ================================================================================

@app.route('/')
def index():

    return render_template('index.html', appName="Tumor Diseases Classification")


#  Skin Classification Diseases
@app.route('/predict_tumor_Api', methods=["POST"])
def skin_api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        print("image loaded....") 
        image_arr= img_preprossing(image)
        print("predicting ...")
        new_predict = tumor_model.predict(image_arr)
        new_predict = np.argmax(new_predict)
    
        classes = {
    
            0: 'Cyst',
            1: 'Normal',
            2: 'Stone',
            3: 'Tumor',
        }

        print(classes[new_predict])

        prediction = classes[new_predict]
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict_tumor_disease', methods=['GET', 'POST'])
def skin_predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        img = request.files['fileup']
        print("image loaded....")
        image_arr= img_preprossing(img)
        print("predicting ...")
        new_predict = tumor_model.predict(image_arr)
        new_predict = np.argmax(new_predict)
    
        classes = {
    
            0: 'Cyst',
            1: 'Normal',
            2: 'Stone',
            3: 'Tumor',
        }

        print(classes[new_predict])

        prediction = classes[new_predict]

        return render_template('index.html', prediction=prediction, appName="Tumor Diseases Classification")
    else:
        return render_template('index.html',appName="Tumor Diseases Classification")


if __name__ == '__main__':
    app.run(debug=True)


