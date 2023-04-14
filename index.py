from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib


app = Flask(__name__)

# initial Route Path ['/'] then on index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route Path['/predict] then on result.html 
@app.route('/predict', methods=['POST'])
def predict():
    # Process data from form client 
    data = pd.DataFrame(request.form, index=[0])

    data = data.astype(int)

    # Load model saved at file model.py
    model = load_model()

    # prediction result 
    prediction = model.predict(data)

    # Render result at result.html with paramater is prediction
    return render_template('result.html', prediction=prediction[0])

# function load_model() load model from file model.pkl via joblib.load('model.pkl')
def load_model():
    # model = KNeighborsClassifier(n_neighbors=3)
    model = joblib.load('model.pkl')
    return model

if __name__ == '__main__':
    app.run(debug=True)
