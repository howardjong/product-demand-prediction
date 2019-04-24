import numpy as np
import pandas as pd
import pickle, traceback
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if rfr:
        try:
            json_ = request.json
            print(json_)

            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=['Product_Code', 'Year', 'Month', 'Day', 'Dayofweek'],
                fill_value=0)
            
            prediction = list(rfr.predict(query))
            
            return jsonify({'prediction':str(prediction)})

        except:
            return jsonify({'trace':traceback.format_exc()})

        else:
            print('Train the model first.')
            return('No model here to use.')
            

if __name__ == '__main__':
 
    MODEL_FILE_NAME = 'model.pkl'

    with open(MODEL_FILE_NAME, 'rb') as data:
        rfr = pickle.load(data)
    print('Model loaded.')

	# to prevent from having to restart the server everytime while developing
    app.run(debug=True)