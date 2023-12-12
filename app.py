from flask import Flask
from flask import request
from flask import jsonify
import numpy
import pickle
from sklearn.feature_extraction import DictVectorizer
from xgboost import DMatrix

with open('model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.pkl', 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('ev_price')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = dv.transform(data)
    X_DMatrix = DMatrix(X, feature_names = dv.feature_names_)

    y_pred = round(model.predict(X_DMatrix)[0])

    price = numpy.exp(y_pred).round(2)
    result = {"price": price}

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=1616)