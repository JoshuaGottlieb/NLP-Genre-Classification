# Classifying Music Genres Based on Song Lyrics

**Author**: Joshua Gottlieb

## Overview and Business Problem

Classifying music genres is a difficult task. Genre labels are a useful metric to use to group certain types of songs together, but they also are a bit nebulous and inconsistent. Assigning a song to a genre is more of an art than a science, and acquiring accurate genre labels for songs frequently requires extensive expert analysis.

In this project, I wanted to determine if it was possible to accurately assign genre labels to songs based only upon the lyrics present in the song. On the one hand, there are some genres with very generic lyrics, such as Pop, but on the other hand, there are some genres that exhibit unique themes in their songs, such as Death Metal. In theory, it should be possible to determine genre with some degree of accuracy using just NLP techniques on the song lyrics.

## Data Collection

All song metadata (title, artist, album, and genre) were pulled from [musiXmatch.com's developer API](https://developer.musixmatch.com/). This site was initially chosen because it touted having millions of song lyrics available. It turns out this is only partially true - a free API key allows the user to download roughly 30% of the lyrics for each song due to copyright reasons. I tried a few other APIs, but musiXmatch's API was still the best for retrieving song metadata, as it allowed me to query by genre, with flags set to match songs with lyrics specifically in English. Many of the other APIs had far less organization, and so I retrieved 10,000 song titles for each of 12 genres:

<ul>
    <li> Alternative </li>
    <li> Black/Death Metal </li>
    <li> Blues </li>
    <li> Christian Gospel </li>
    <li> Country </li>
    <li> Dance </li>
    <li> Hard Rock </li>
    <li> Jazz </li>
    <li> Hip Hop/Rap </li>
    <li> Pop </li>
    <li> R&amp;B/Soul </li>
    <li> Reggae </li>
</ul>

These genres were picked somewhat arbitrarily. I inspected the available genres from musiXmatch, as well as the genres/subgenres with large amounts of data available, and these genres were the ones which stood out the most. It should be noted that Christian Gospel contained Christian Rock underneath it, which is closer to the various rock genres than to gospel music, as far as I am concerned. I also did not apply any rules to prevent the acquisition of duplicates, if they appeared in multiple genres. The musiXmatch API has a limit of about 2,000 calls per day, meaning it took about 6 days to obtain the desired data.

Lyrics were scraped from [SongLyrics](https://www.songlyrics.com/) using [requests](https://requests.readthedocs.io/en/latest/) and [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/). Lyrics data was then stitched with the song metadata using artist/title pairs as the join keys, in preparation for cleaning. It should be noted that even when running the scraping operation on two machines simultaneously, it took roughly 3 days to scrape the data, due to interruptions caused by bad URL construction due to song titles that were formatted in an irregular manner. All raw data can be found in zipped format in [./data/raw].

## Data Cleaning

Processing my scraped song lyrics (each song will be called a document and the entire list of documents will be referenced as a corpus for the rest of this README) required some effort and careful thought. First, I had to trim the corpus for actual text preprocessing. Some documents had errors from scraping, these were dropped. Each document was split into a list of strings for further preprocessing. Then, any documents which did not actually contain song lyrics were dropped, which was possible since songlyrics.com has specific wording for songs for which they do not have lyrics.

Further preprocessing involved using a series of [RegEx](https://docs.python.org/3/library/re.html) transformations to remove leading phrases that were scraping artifacts, remove all non-alphabetic characters, remove all words (henceforth referred to as tokens) of 2 or fewer characters, and strip white spaces. Stop words from [NLTK's stopword list](https://pythonspot.com/nltk-stop-words/) were removed, and each document was converted to a list of tokens. Finally, each token was then lemmatized using [spaCy's core English pipeline](https://spacy.io/models/en). To see the full list of transformations used, see [the scraping module](./notebooks/project_functions/scraping.py).

After cleaning, duplicates were dropped, and all records for Hip Hop/Rap and R&B/Soul were dropped, as each genre had fewer than 3000 documents. The document distribution after cleaning was as follows:

| Genre Count                           |
| :-----------------------------------: |
| ![](./visualizations/genre_count.png) |

To be used in models and other tests, it is necessary to vectorize each document in my corpus. I used a variety of different vectorization methods, including [SkLearn's CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), n-gram encoding using [gensim's Phraser](https://radimrehurek.com/gensim/models/phrases.html), and averaged neural network encoding using [word2vec and glove](https://radimrehurek.com/gensim/models/word2vec.html) also from gensim. Note that I did not train a neural network myself, and so I simply used the pre-trained models to vectorize each token, then averaged all tokens in the document to get vectorized documents, which is not the best method for using these pre-trained neural networks but was something I could use in a shallow learning model.

## Topic Modeling and Data Exploration

Before throwing the data into some models, I decided to do some topic modeling to see if there was a natural representation of the genres that was different from the current labels that exist in my data set. Below are the results of the coherence score analysis that was performed.

| LDA Coherence Scores Across N-Topics                       |
| :--------------------------------------------------------: |
| ![](./visualizations/coherences.png)                       |

As can be seen, the coherence scores are highest at 8 topics, a different number than the 10 genres remaining in our dataset.

| pyLDAvis for LDA on Gensim Bigrams with 8 Topics           |
| :--------------------------------------------------------: |
| ![](./visualizations/pyLDAvis_lda_8_topics.png)            |

The result of the 8 topic LDA model creates clear clusters representing Christian Gospel (topic 7), Death/Black Metal (topic 5), and Reggae (topic 8), but the rest of the topics are not clearly identifiable as any particular genre and contain generic words that could be in anything. In addition, topic 4 is completely contained in topic 2, meaning that we might have subgenres instead of distinct genres. This is an indication that our genres might not be very separable.

Next, I performed a TSNE analysis to see how clumped the data was.

| TSNE Gensim 4-Gram Encoded Train Set Ground-Truth Labels   |
| :--------------------------------------------------------: |
| ![](./visualizations/TSNE_gensim_4gram_train.png)          |

This matches with the results seen in the pyLDAvis results - there is a cluster on the left for Christian Gospel, a cluster in the middle for Death/Black Metal, and a cluster on the right for Reggae, but everything else is jumbled together with no clear separation. We should expect poor results from our models.


## Results

I ran a variety of models, including an [SkLearn Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) classifier, and a set of untuned and tuned [XGBoosted Tree](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier) classifiers, using all of the vectorization methods mentioned before. It was not possible to use Multinomial Naive Bayes on the glove or word2vec vectorizations, as the document vectors contained negative values, and the CountVectorizer vectorizations performed so poorly compared to the Gensim vectorizations that I decided not to run them through the XGBoost models.

| Model Scores                                               |
| :--------------------------------------------------------: |
| ![](./visualizations/model_scores.png)                     |

The best model is the Gensim 4-gram vectorization using a tuned XGBoosted Tree. It should be noted that the tree was tuned to optimize the Gensim bigram vectorization, as I did not have the time or resources to individually tune to each vectorization. The tuned XGBoost used 1000 estimators, a max depth of 3, and a gamma of 0.1, with other parameters at their default. The XGBoost took much longer to run than the Multinomial Naive Bayes, so the second best model is the Multinomial Naive Bayes using the Gensim trigram vectorization, in terms of accuracy gained for minimal fit time.

| Best Model Confusion Matrix                                |
| :--------------------------------------------------------: |
| ![](./visualizations/best_model_confusion_matrix.png)      |

Looking at the confusion matrix for the best model reveals few surprises based off of our EDA. Death/Black Metal, Christian Gospel, and Reggae are very separable, while the rest of the genres perform poorly. In particular, Pop, Alternative, Dance, and Hard Rock are extremely prone to miscategorization. Death/Black Metal is sometimes miscategorized as Alternative or Hard Rock and vice-versa, which makes sense Death/Black Metal is a more extreme subgenre of the overall Rock genre, Hard Rock is also a subgenre of Rock, and Rock shares a lot of features with Alternative.

| Best Model Classification Report                           |
| :--------------------------------------------------------: |
| ![](./visualizations/best_model_classification_report.png) |

Analyzing the classification report clarifies prediction quality by class. Reggae appears to live in its own island - its very high precision indicates that few genres are misclassified as Reggae. Death/Black Metal and Christian Gospel perform well across the board. Alternative and Pop are extremely generic genres, so it makes sense that they would not be distinct enough to be identified by lyrics.

| TSNE Gensim 4-Gram Encoded Documents Test Predictions      |
| :--------------------------------------------------------: |
| ![](./visualizations/TSNE_gensim_4gram_test.png)           |

Finally, examining the predicted labels on the test set TSNE reaffirms the patterns we saw during EDA. Christian Gospel, Death/Black Metal, and Reggae still have their own clusters on the left, in the middle, and on the right respectively, while everything else is jumbled together.

## Conclusions

In general, it is quite difficult to classify song genres solely by lyrics. Some genres do exhibit unique vocabularies that allow them to be easily identified, while others are quite generic and fail to stand out. NLP is a powerful tool, but in the end, it is probably easier to classify song genres based on more musical qualities, such as instruments used, tempo/beat, musical key, and so on.

## Next Steps

Next steps include:
<ul>
    <li>Gather more data, as wrangling the musiXmatch API and web scraping produced limited results.</li>
    <li>Combine and drop genres in accordance with topic modeling and domain knowledge.</li>
    <li>Turn the problem on its head and see if there are songs which can be "genre-swapped" due to having lyrics which could be in multiple genres.</li>
</ul>

## For More Information

Please look at my full analysis in [Jupyter Notebooks](./notebooks), or in my [presentation](./presentation/Identifying_Exoplanets_Using_Machine_Learning.pdf), and code in the [Project Modules](./notebooks/project_functions).

For any additional questions, please contact: **Joshua Gottlieb (joshuadavidgottlieb@gmail.com)**

## Repository Structure

```
├── README.md                               <- The top-level README for reviewers of this project   
├── .gitignore                              <- Hidden file specifying which files to ignore
├── notebooks                               <- Folder containing Jupyter notebooks with project code
│   ├── Data-Collection.ipynb
│   ├── Data-Cleaning-and-Text-Preprocessing.ipynb
│   ├── EDA.ipynb
│   ├── Modeling.ipynb
│   ├── Visualizations.ipynb
│   ├── project_functions                   <- Subfolder acting as Python module holding submodules with functions
│   │   ├── data_cleaning.py
│   │   ├── EDA.py
│   │   ├── __init__.py
│   │   ├── modeling.py
│   │   ├── scraping.py
│   │   ├── utils.py
│   │   └── visualizations.py
│   └── Keys.py                             <- Python file ignored by .gitignore holding API keys
├── data                                    <- Folder containing external and code-generated data
│   ├── raw                                 <- Subfolder containing raw data pulled from APIs and webscraping
│   ├── cleaned                             <- Subfolder containing cleaned data used by notebooks
│   │   ├── test                            <- Subfolder containing test data
│   │   └── train                           <- Subfolder containing train data
│   ├── gensim                              <- Subfolder containing pickled fitted gensim objects such as dictionaries
│   ├── models                              <- Subfolder containing pickled fitted models
│   └── tsne                                <- Subfolder containing pickled dataframes containing TSNE information
├── presentation                            <- Folder containing PDF of presentation
└── visualizations                          <- Folder containing PDF of presentation
    └── pyLDAvis                            <- Subfolder containing saved html files of pyLDAvis objects
```