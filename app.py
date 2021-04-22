from flask import Flask, jsonify, request
import pickle as pickle
import math
from scipy.sparse import hstack
import json
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
app = Flask(__name__)
nltk.download('vader_lexicon')

@app.route('/')
def index():
    return "Hello"


@app.route('/PredictLikes', methods=["GET"])
def profile():
    try:
        json_data = request.json
        categoryId = json_data["categoryId"]
        view_count = json_data["view_count"]
        video_count = json_data["video_count"]
        subscriber_count = json_data["subscriber_count"]
        video_title = json_data["video_title"]
        description = json_data["description"]
        likes = predict_likes(categoryId, view_count, video_count, subscriber_count, video_title, description)
        x = {
            "likes": likes
        }
        y = json.dumps(x)
        y = json.loads(y)
        return jsonify(y)
    except TypeError :
        return "No data found"



def find_sentiment(title, description, sid):
    try:
        ss = sid.polarity_scores(title)
        ss2 = sid.polarity_scores(description)

        title_sentiment = ss['compound']
        description_sentiment = ss2['compound']
        
        # print(title_sentiment)
        # print(description_sentiment)
        
        return [title_sentiment, description_sentiment]
    except Exception as e:
        print (e)
        pass

def predict_likes (categoryId, view_count, video_count, subscriber_count,video_titile, description):
    sid = SentimentIntensityAnalyzer()
    list_of_sentiment = find_sentiment(video_titile, description,sid)
    title_sentiment = list_of_sentiment[0]
    description_sentiment = list_of_sentiment[1]

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
        title_sentiment,
        description_sentiment,
    ]) 

    categorical_features = []
    categorical_features.append(["None"])
    bias = ""
    if (subscriber_count < 5*10**5):
        bias = "0"
    elif (subscriber_count >= 5*10**5 and subscriber_count < 5*10**6):
        bias = "1"
    else:
        bias = "2"
    

    with open("Models/Train" + bias + "/my_numerical_encoder.pkl", 'rb') as fid:
        numencoder = pickle.load(fid)
    with open("Models/Train" + bias + "/my_categorical_encoder.pkl", 'rb') as fid:
        catencoder = pickle.load(fid)


    numerical_features2 = numencoder.transform(numerical_features)

    categorical_features2 = catencoder.transform(categorical_features)

    X = hstack([numerical_features2, categorical_features2])


    with open("Models/Train" + bias + "/my_dumped_classifier.pkl", 'rb') as fid:
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


if __name__ == "__main__":
    app.run()
