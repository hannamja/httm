import datetime
import json
import random

from flask import Flask, request
from CNN.m_predict import *
from decision_tree import *
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    token_label1, sam_label1 = load_data(url_full_train_data, "Sheet1", 300)
    data = request.form.get('answer_mood')
    label1_pred = predict(data, token_label1, sam_label1, load_aspect_model('D:\\model\\CNN_train_3c_relu.json',
                                                                            'D:\\HDH\\dts-phuclong_raw_train_2c-001-0.0144-1.0000.h5'),
                          labels)
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    if label1_pred[0] == 'nev':
        mood = 1
    else:
        mood = 0
    random.seed(datetime.datetime.now())
    print(datetime.datetime.now())
    pred_value = [1, random.random(), random.random(), random.choice([1, 2, 3, 4])]
    print(pred_value)
    y_pred_entropy = clf_entropy.predict([[1, random.random(), random.random(), random.choice([1, 2, 3, 4])]])

    print(y_pred_entropy)
    print(label1_pred)
    mood_ = ""
    if mood == 0:
        mood_ = "Tích cực"
    else:
        mood_ = "Không tích cực"

    return json.dumps({"name": y_pred_entropy[0], 'mood': mood_})


app.run()
