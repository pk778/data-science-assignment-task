import pickle
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from flask import Flask, request
from train_model import add_cat_flag_cols, clean_text

logging.getLogger().setLevel(logging.INFO)
app = Flask(__name__)


def read_file(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file


cat_encoder = read_file('model_files/cat_encoder.pk')
tfidf = read_file('model_files/tfidf.pk')
selector = read_file('model_files/selector.pk')
categories = read_file('model_files/categories.pk')

model = tf.keras.models.load_model('model_files/my_tf_model')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


@app.route('/predict', methods=['POST'])
def predict():
    """returns category prediction and corresponding probability"""
    d = request.json
    data = pd.DataFrame.from_dict({'title': [d.get('title')],
                                   'text':  [d.get('text')],
                                   'url':  [d.get('url')]})
    data['title_text'] = data['title'] + ' ' + data['text']
    data['title_text_clean'] = data.apply(lambda x: clean_text(x.title_text, lemmatizer, stop_words), axis=1)
    data = add_cat_flag_cols(data, categories)

    tfidf_scores = tfidf.transform(data.title_text_clean)
    tfidf_topk = selector.transform(tfidf_scores)
    tfidf_topk_input = tfidf_topk.toarray()
    cat_cols = [cat for cat in data.columns if cat in categories]
    url_cat_input = np.array(data[cat_cols].values)

    preds = model.predict([tfidf_topk_input, url_cat_input])
    preds = [[preds[i].argmax(), preds[i][preds[i].argmax()]] for i in range(len(preds))]
    preds = [[cat_encoder.classes_[pred[0]], pred[1]] for pred in preds]

    data['category_pred'] = [pred[0] for pred in preds]
    data['pred_prob'] = [pred[1] for pred in preds]

    return {"category_pred": str(data['category_pred'].values),"pred_prob":str(data['pred_prob'].values)}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
