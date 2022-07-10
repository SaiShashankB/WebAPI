import numpy as np
from flask import Flask, request, render_template
import pickle
import model.pkl
import scaler.pkl

app = Flask(__name__)

model1 = pickle.load(open("model.pkl", "rb"))
scale = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    transformed_features = scale.transform(features)
    prediction = model1.predict(transformed_features)

    return render_template("index.html", prediction_text="The ad is {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
