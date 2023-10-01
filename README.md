# Housing-ML

## 1. Rental Price Development

[**Price forecast notebook viewer**](https://nbviewer.org/github/andreicap/housing-ml/blob/main/price_forecast.ipynb)

**Summary**: Building a pricing engine for housing market.

There are 2 approaches tried: regression and time series forecasting.

### Regression
The regression predictor is meant to be used inline to predict the price of an estate, e.g. by an user introducing the required parameters (space size, type of buidling etc.)

The forectasting is done using `Pycaret`, for automatized model selection and feature engineering.

Potential improvements:
* add more data (both samples and features), possibly from other sources
* improve the current feature extraction
* try more methods/models
* use a gridsearch for hyperparameter tuning
* determine the metric that presents the most interest to the business/user and optimize for that


### Time Series Forecasting
The time series forecasting is meant to be used both in-house and by potential clients, for future insights for better decision-making.

The forectasting is done using `Pycaret`, for automatized model selection and feature engineering.

Potential improvements:
* try multivariate forecasting
* try prediction LSTM/RNN for forecasting (requires more samples/features)

## 2. Review-based User Recommendations

[**Recommendations notebook viewer**](https://nbviewer.org/github/andreicap/housing-ml/blob/main/recommendations.ipynb)

The work is split in 2 parts: 
1. Sentiment extraction from user reviews
2. Recommender System

### Sentiment extraction from user reviews

In this part, user comments are cleaned with `pandas` and python functions, tagged with a language using `langdetect` and the sentiment is extracted using `nltk`.

The sentiment analyzer returns a score from -1 to 1, representing the *rating* of the listing by a user.

Potential improvements:
* use ChatGPT/LLMs for comment tagging (example provided in the notebook ChatGPT Tagging section). More granular sentiment per topic ca be extracted (e.g. a user might like the cleaniness but no the location)
* use a faster language detection algorithm and/or parallelize the run (took 1h to run)

### Recommender System

The recommender system is a comparison for different implementations of model based collaborative filtering.
I eneded up using the `surprise` package, after trying  also the `SVDS` from `scikit-learn` and `FUNK_SVD` packages.

It compares 2 models: `KNNWithMeans` and `SVD`. Further testing is required to determined the better model (for this test set they were very similar).

Potential improvements:
* try more methods for recommender systems
* try classical ML for ranking 
* user a more powerful machine that can run on the whole dataset

## Next Steps
* move the code from jupyter notebooks to python files (for deployment)
* build inferrence pipeline (model consumption)
* build the respective dashboard/interfaces to make the results accessible by everyone


## Running instructions:

1. Create a new and enviroment, and run:
```bash
pip install -r requirements.txt
```

2. Download and unzip  the `recommendations.zip` and `forecasting.zip` inside the data folder.

3. Then you can run the 2 jupyter notebooks with `jupyter notebook` command.

4. Run `mlflow ui` in the terminal to see the history of model comparisons and runs (for the price forecasting only)





