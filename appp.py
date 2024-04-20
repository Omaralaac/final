from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
#creating an Api object
api = Api(app)

#prediction Api call
model = joblib.load(open('D:\\CS Materials\\Project\\PData\\api\\you\\model.pkl','rb'))

@app.route('/')
def home():
    return 'السلام عليكم ورحمة الله وبركاته'

@app.route("/predict", methods=["POST"])
def predict():
    rates = request.json
    query_df = pd.DataFrame(rates)
    prediction = model.predict(query_df)
    return jsonify(list(prediction))




if __name__ == '__main__':
    app.run(debug = True)


    