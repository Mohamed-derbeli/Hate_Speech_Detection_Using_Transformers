from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


app = Flask(__name__)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Required functions:

def clean_txt(tweet):
    regex_pat = re.compile(r'\s+')
    Tweet = re.sub(regex_pat," ",tweet)
    Tweet= re.sub(r'@[\w\-]+',"",Tweet)
    url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    Tweet = re.sub(url_regex,"",Tweet)
    Tweet = nltk.word_tokenize(Tweet)
    Tweet = [w.lower() for w in Tweet]
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    Tweet = [re_punc.sub('', w) for w in Tweet]
    Tweet = [word for word in Tweet if word.isalpha()]
    Tweet = [w for w in Tweet if w not in  stopwords] 
    Tweet = [word for word in Tweet if len(word) > 2]
    lemmatizer = WordNetLemmatizer() 
    lem_words = [lemmatizer.lemmatize(word) for word in Tweet]
    combined_text = ' '.join(lem_words)
    return combined_text


def make_prediction(tweet):
    cleaned_text = clean_txt(tweet)
    lst = []
    lst.append(cleaned_text)
    vectorized = pickle.load(open("model/vectorize_tweet.pkl", "rb"))
    vectorized = vectorized(lst)
    model = pickle.load(open("models/model_RF.pkl", "rb"))
    pred = model.predict(vectorized)
    if int(pred)==0:
        return 'free speech'
    else:
        return 'hate speech' 

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')


def predict():
    tweet= request.form['Input']  #.values()
    print(tweet)
    output= make_prediction(tweet)
    return render_template('predictor.html', prediction_text='Detection: {}'.format(output))


if __name__=="__main__":
    # For local development:
    # app.run(debug=True)
    # For public web serving:
    app.run(host='0.0.0.0')
    app.run()
