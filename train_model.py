import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import logging

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

logging.getLogger().setLevel(logging.INFO)
SEED=4


def remove_special_chars(x):
    return re.sub(r"[^A-Za-z0-9]+", " ", x)


def remove_extra_spaces(x):
    return " ".join(x.split())


def remove_stopwords(text, stop_words):
    words = text.split(' ')
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)


def lower_text(x):
    return x.lower()


def lemmatize_words(text, lemmatizer):
    words = text.split(' ')
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)


def clean_text(x, lemmatizer, stop_words):
    x = remove_special_chars(x)
    x = remove_extra_spaces(x)
    x = lower_text(x)
    x = remove_stopwords(x, stop_words)
    x = lemmatize_words(x, lemmatizer)
    return x


def add_cat_flag_cols(data, categories):
    "add 1/0 columns if category as word(s) is/are included in url or not"
    categories = [(cat, cat.split('_')) for cat in categories]
    categories = {cat[0]: cat[1] for cat in categories}

    for cat in categories:
        data[cat] = data['url'].str.contains('|'.join(categories[cat])).astype(int)

    return data


def create_model():
    """
    neural network with tensorflow functional api to concatenate values
    from tfidf pipeline and url flag values
    """
    tfidf_layer = tf.keras.layers.Input(shape=(300,))
    url_flag_layer = tf.keras.layers.Input(shape=(12,))
    cc = tf.keras.layers.Concatenate()([tfidf_layer,url_flag_layer])
    cc_f = tf.keras.layers.Flatten()(cc)
    cc1 = tf.keras.layers.Dense(100, activation='relu')(cc_f)
    cc2 = tf.keras.layers.Dropout(0.2)(cc1)
    cc3 = tf.keras.layers.Dense(50, activation='relu')(cc2)
    output_layer = tf.keras.layers.Dense(12,activation="softmax")(cc3)
    model = tf.keras.Model(inputs=[tfidf_layer, url_flag_layer], outputs=output_layer)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),metrics="accuracy")
    return model


def write_file(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)


if __name__ == '__main__':
    logging.info('loading and preparing data')
    data = pd.read_csv('data_redacted.tsv', sep='\t')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    data['title_text'] = data['title'] + ' ' + data['text']
    data['title_text_clean'] = data.apply(lambda x: clean_text(x.title_text, lemmatizer, stop_words), axis=1)

    categories = list(data['category'].unique())
    data = add_cat_flag_cols(data, categories)

    X_train, X_test = train_test_split(data, test_size=0.2, random_state=4, stratify=data['category'])

    # Calculation of tfidf scores and selection of top 300 related to target column, in our case 'category'
    tfidf = TfidfVectorizer(min_df=10, max_df=0.3, analyzer='word', max_features=3000)
    title_text_clean_tfidf_scores = tfidf.fit_transform(X_train.title_text_clean)
    selector = SelectKBest(f_classif, k=300)
    title_text_clean_tfidf_topk = selector.fit_transform(title_text_clean_tfidf_scores, X_train['category'])

    tfidf_topk_input = title_text_clean_tfidf_topk.toarray()

    # prepare input based on category flag columns for url
    cat_cols = [cat for cat in X_train.columns if cat in set(X_train['category'])]
    url_cat_input = np.array(X_train[cat_cols].values)

    #create encodings for the categories available
    cat_encoder = LabelBinarizer()
    cat_encodings = cat_encoder.fit_transform(X_train['category'])

    tf.random.set_seed(SEED)

    model = create_model()

    logging.info('model training')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=2)
    model.fit([tfidf_topk_input, url_cat_input], cat_encodings, batch_size=20, epochs=20, validation_split=0.1,
              callbacks=[callback], verbose=False)

    logging.info('saving tf model')
    model.save('model_files/my_tf_model')

    logging.info('saving tfidf vectorizer')
    write_file('model_files/tfidf.pk', tfidf)

    logging.info('saving topk selector')
    write_file('model_files/selector.pk', selector)

    logging.info('saving encoder')
    write_file('model_files/cat_encoder.pk', cat_encoder)

    logging.info('saving list of available categories')
    write_file('model_files/categories.pk', categories)
