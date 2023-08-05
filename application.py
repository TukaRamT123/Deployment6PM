# Libraries imported
from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initilize the flask application
application = Flask(__name__)
app = application

# Create the homepage
@app.route('/')
def home_page():
    return render_template('index.html')

# Create the logic for prediction
@app.route('/predict',methods=['POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Load all the pickle files
        with open('notebooks/LabelEnc.pkl','rb') as file1:
            le = pickle.load(file1)
        
        with open('notebooks/Scaler.pkl','rb') as file2:
            scaler = pickle.load(file2)

        with open('notebooks/best_knn.pkl','rb') as file3:
            model = pickle.load(file3)

        # Take all user input from the webpage
        sep_len = float(request.form.get('sepal_length'))
        sep_wid = float(request.form.get('sepal_width'))
        pet_len = float(request.form.get('petal_length'))
        pet_wid = float(request.form.get('petal_width'))

        # Convert values to dataframe
        xnew = pd.DataFrame([sep_len, sep_wid, pet_len, pet_wid]).T
        xnew.columns =['sepal_length','sepal_width','petal_length','petal_width']

        # Perform scaling on dataframe
        xnew_pre = pd.DataFrame(scaler.transform(xnew), columns=xnew.columns)

        # Perform predictions
        pred = model.predict(xnew_pre)

        # Predicted Label
        pred_lb = le.inverse_transform(pred)[0]

        # probability
        prob = model.predict_proba(xnew_pre).max()

        # Prediction string
        prediction = f'{pred_lb} with Probability {prob:.4f}'

        return render_template('index.html', prediction=prediction)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0')