## Instructions

Results for the solution are included in *solution.html*. 

Below you can follow the instructions so as to reproduce the results with jupyter notebook or
train and containerize the model so as to make post requests.

Before going forward with the instructions below, make sure that you have the 
**data_redacted.tsv** available in the repo's directory.

## Run with Jupyter notebook

To run the solution you can use the conda environment manager and follow the steps below:

1. Create & activate enviroment, install dependencies and jupyter notebook by executing 
following commands:

    ```
    conda create -n upday python=3.8
    conda activate upday
    
    pip install -r requirements.txt
    python -m nltk.downloader stopwords
    python -m nltk.downloader wordnet
    python -m nltk.downloader omw-1.4
    
    pip install notebook  
    ```

2. Open jupyter notebook with following command and run *solution.ipynb*.
  
    ```
    jupyter notebook
    ```


## Run with docker 

1. To build the docker image with name *serve-article-cat* execute the following: 

    ```
    docker build -t serve-article-cat .
    ```

2. To run a container from this image run the following:

    ```
    docker run -p 8080:8080 serve-article-cat python3 app.py
    ```

3. Container now is up and running, you can get predicted category and corresponding 
    probability for an item by making requests as following:

    ```
    curl -X POST localhost:8080/predict -d '{"title":"news item title","text":"news item text","url":"www.newsitemurl.com"}' -H 'Content-Type: application/json'
    ```
