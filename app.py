import pickle as pickle
import math
from scipy.sparse import hstack
from flask import Flask
app = Flask(__name__)

import json



@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/categoryId/<categoryId>/view_count/<view_count>/video_count/<video_count>/subscriber_count/<subscriber_count>')
def profile(categoryId, view_count, video_count, subscriber_count):
    likes = predict_likes(categoryId, view_count, video_count, subscriber_count)
    x = {
    "likes": likes
    }

    # convert into JSON:
    y = json.dumps(x)

    # the result is a JSON string:
    print(y)
    return y

def predict_likes (categoryId, view_count, video_count, subscriber_count):



    numerical_features = []
    categorical_features = []

    # categoryId = int(input("Enter categoryID: "))
    duration = 1
    # view_count = int(input("Enter views count of channel: "))

    # video_count = int(input("Enter video count of channel:" ))
    # subscriber_count = int(input("Enter subscriber count of channel: "))


    numerical_features = []    
    numerical_features.append([
        categoryId,
        duration,
        view_count,
        video_count,
        subscriber_count,
    ]) 

    categorical_features = []
    categorical_features.append(["None"])

    with open('Models/my_numerical_encoder.pkl', 'rb') as fid:
        numencoder = pickle.load(fid)
    with open('Models/my_categorical_encoder.pkl', 'rb') as fid:
        catencoder = pickle.load(fid)


    numerical_features2 = numencoder.transform(numerical_features)

    categorical_features2 = catencoder.transform(categorical_features)

    X = hstack([numerical_features2, categorical_features2])


    with open('Models/my_dumped_classifier.pkl', 'rb') as fid:
        model = pickle.load(fid)
        
    y_pred = [] 
    y_pred = model.predict(X)
    y_lower = math.floor(y_pred)
    y_upper = math.ceil(y_pred)
    y_pred_lower  = 10**y_lower
    y_pred_upper  = 10**y_upper

    if (y_upper == y_lower):
        print(y_pred_lower)
    else:
        print(y_pred_lower, y_pred_upper)

    print ('Predicted : ', int(10**y_pred))
    return int(10**y_pred)