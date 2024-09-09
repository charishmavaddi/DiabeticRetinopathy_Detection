from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'C:/Users/nikhitha/Desktop/Diabetic_Retinopathy_Detection_R10/Tensorflow/'



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    import numpy as np
    from tensorflow.keras.models import load_model
    import cv2
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical
    labelEncoder = LabelEncoder()
    model = load_model('tensorflow_model.h5')
    class_labels = ['0', '1', '2', '3', '4']
    file_name = request.files['image'].filename
    image_file = request.files['image']
    print(file_name)
    X=[]
    file_bytes = np.asarray(bytearray(image_file.read()), dtype= np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (300,300))
    #img = np.expand_dims(img, axis=0)
    X.append(np.array(img).reshape(1, 300, 300, 3))
    predictions = model.predict(X)
    predicted_labels = [class_labels[np.argmax(p)] for p in predictions]
    text = str(predicted_labels[0])

    if text == '0':
        text = "No DR"
    elif text == '1':
        text = "Mild"
    elif text == '2':
        text = "Moderate"
    elif text == '3':
        text = "Severe"
    elif text == '4':
        text = "Proliferative DR"
    else :
        text = "No DR"

    return render_template('result.html', text=text)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host = "0.0.0.0", port = 80, debug=True)
