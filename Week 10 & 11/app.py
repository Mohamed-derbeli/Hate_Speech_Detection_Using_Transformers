from flask import Flask, request, render_template
import pickle
import string
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from waitress import serve
import random


app = Flask(__name__)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Required functions:

def clean_txt(tweet):
    regex_pat = re.compile(r'\s+')
    tweet = re.sub(regex_pat," ",tweet)
    tweet= re.sub(r'@[\w\-]+',"",tweet)
    url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweet = re.sub(url_regex,"",tweet)
    tweet = nltk.word_tokenize(tweet)
    tweet = [w.lower() for w in tweet]
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tweet = [re_punc.sub('', w) for w in tweet]
    tweet = [word for word in tweet if word.isalpha()]
    tweet = [w for w in tweet if w not in stopwords]
    tweet = [word for word in tweet if len(word) > 2]
    lemma_tizer = WordNetLemmatizer()
    lem_words = [lemma_tizer.lemmatize(word) for word in tweet]
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
    # detect= " "
    # if int(pred)==0:
    #     detect= 'free speech'
    # else:
    #     detect= 'hate speech'
    return pred

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('predictor.html')


@app.route('/predict',methods=['POST'])
def predict():
    tweet= request.form['Input']
    print(tweet)
    output= make_prediction(tweet)
    print(output)
    return render_template('predictor.html', prediction_text='Detection: {}'.format(output))



if __name__ == "__main__":
    # serve(app, host="0.0.0.0", port=8845)
    # app.debug=True
    # app.run()

    port = 5000 + random.randint(0, 999)
    print(port)
    url = "http://127.0.0.1:{0}".format(port)
    print(url)
    app.run(use_reloader=False, debug=True, port=port)
