#Base image
FROM python:3.8.3-slim-buster

#Copy files
COPY . /src

#Install dependencies
RUN pip install -r /src/requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

#change directory
WORKDIR /src

#Train model
RUN python train_model.py
