import pandas as pd
import numpy as np
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')

# defining data processing function
def preprocess_string(text):

    # every char except those in the English alphabets is replaced
    processed_string = re.sub('[^a-z\s]+',' ', text, flags=re.IGNORECASE)

    # multiple spaces are replaced by single space
    processed_string = re.sub('(\s+)',' ', processed_string)

    # converting the processed string to lower case
    processed_string = processed_string.lower()

    # return the processed/stemmed/cleaned string
    return processed_string

# Calculating how accurate the model's prediction is
def calc_accuracy(true_label, pred_label):
    no_true_labels = len(true_label)
    no_matched_labels = np.sum(true_label == pred_label)
    acc = no_matched_labels / no_true_labels
    return acc

class NaiveBayes:

    # initialize the instance and define the number of unique classes
    def __init__(self, num_unique_classes):

        self.classes = num_unique_classes

    # splits the text using space as a tokenizer
    # adds every tokenized word to its corresponding dictionary Bag of Word
    # bow_index: shows which Bag of Word category this text belongs to
    def addToBagOfWords(self, text, bow_index):

        if isinstance(text, np.ndarray):
            text = text[0]

        # for every word in preprocessed example
        for token_word in text.split():

            # increment the number of tokenized words
            self.bagOfWords_dicts[bow_index][token_word] = self.bagOfWords_dicts[bow_index][token_word] + 1

    # training function for the NB Model
    def train(self, dataset, labels):

        self.data = dataset
        self.labels = labels
        self.bagOfWords_dicts = np.array([defaultdict(lambda: 0) for index in range(self.classes.shape[0])])

        # only convert data to numpy arrays if initially not passed as numpy arrays
        if not isinstance(self.data, np.ndarray):
            data = np.array(self.data)

        # only convert labels to numpy arrays if initially not passed as numpy arrays
        if not isinstance(self.labels, np.ndarray):
            labels = np.array(self.labels)

        # constructing a bag of words for each category
        for category_index, category in enumerate(self.classes):

            # filter all texts of category == category
            all_category_examples = self.data[self.labels == category]

            # get texts processed, stemmed/cleaned
            prepro_texts = [preprocess_string(category_example) for category_example in all_category_examples]
            prepro_texts = pd.DataFrame(data=prepro_texts)

            # make a bag of words of this particular category
            np.apply_along_axis(self.addToBagOfWords, 1, prepro_texts, category_index)

        probability_classes = np.empty(self.classes.shape[0])
        all_words = []
        category_word_counts = np.empty(self.classes.shape[0])

        for category_index, category in enumerate(self.classes):

            # Calculating prior probability p(c) for each class
            probability_classes[category_index] = np.sum(self.labels == category) / float(self.labels.shape[0])

            # Calculating total counts of all the words of each class
            count = list(self.bagOfWords_dicts[category_index].values())
            category_word_counts[category_index] = np.sum(
                np.array(list(self.bagOfWords_dicts[category_index].values()))) + 1  # |v| is remaining to be added

            # get all words of this category
            all_words += self.bagOfWords_dicts[category_index].keys()

        # combine all words of every category & make them unique to get vocabulary of entire training set
        self.vocab = np.unique(np.array(all_words))
        self.vocab_length = self.vocab.shape[0]

        # computing denominator value
        denom_vals = np.array(
            [category_word_counts[category_index] + self.vocab_length + 1 for category_index,
                                                                              category in enumerate(self.classes)])

        self.categories_information = [(self.bagOfWords_dicts[category_index], probability_classes[category_index],
                           denom_vals[category_index]) for category_index, category in enumerate(self.classes)]

        self.categories_information = np.array(self.categories_information)

    # estimates posterior probability of the given test text
    def getTestTextProbability(self, test_text):

        # store the probability with respect to each class
        likelihood_proba = np.zeros(self.classes.shape[0])

        # finding probability with respect to each class of the given test text
        for category_index, category in enumerate(self.classes):

            # split the test text and get probability of each test word
            for test_token in test_text.split():

                # get total count of this test token from its respective training dictionary to get numerator value
                test_token_counts = self.categories_information[category_index][0].get(test_token, 0) + 1

                # now obtain the likelihood of this test_token word
                test_token_proba = test_token_counts / float(self.categories_information[category_index][2])

                likelihood_proba[category_index] += np.log(test_token_proba)

        # we have likelihood estimate of the given example against every class, but we need posterior probability
        posterior_proba = np.empty(self.classes.shape[0])
        for cat_index, category in enumerate(self.classes):
            posterior_proba[cat_index] = likelihood_proba[cat_index] + np.log(self.categories_information[cat_index][1])

        # return the posterior probability of test text in all unique classes
        return posterior_proba

    # testing function for the NB model
    #  Determines probability of each test text against all classes and predicts the label
    #  against which the class probability is maximum
    def test(self, test_set):

        # stores prediction of each test text
        predictions = []

        for text in test_set:
            # preprocess the test example the same way we did for training set texts
            preproc_text = preprocess_string(text)

            # get the posterior probability of every example
            post_proba = self.getTestTextProbability(preproc_text)

            # pick the maximum value and map against self.classes!
            predictions.append(self.classes[np.argmax(post_proba)])

        #  returns predictions of test texts - A single prediction against every test text
        return np.array(predictions)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    news_dataset = pd.read_csv('news_data.csv')
    print(news_dataset.shape)
    print(news_dataset.head())

    X = news_dataset['text'].values
    y = news_dataset['eval'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=11)
    classes = np.unique(y_train)

    # create a NB model classifier
    nb = NaiveBayes(classes)

    # fit and train the data
    nb.train(X_train, y_train)

    no_classes = nb.test(X_test)
    test_accuracy = calc_accuracy(y_test, no_classes)

    print("Naive Bayes accuracy: ", test_accuracy)

    model_accuracy = {
        'K-NN': 0.8425,
        'NB': 0.9225,
    }

    pd.Series(model_accuracy).plot(kind='bar', color=['red', 'green'])
    plt.ylabel('Accuracy Score')
    plt.ylim((0.0, 1))
    plt.show()

    model_accuracy_basedOnInput = {
        '400': 0.95,
        '500': 0.94,
        '1000': 0.925,
        '1500': 0.953,
        '1750': 0.914,
        '2000': 0.9225,
    }

    pd.Series(model_accuracy_basedOnInput).plot(kind='bar', rot=0, color=['blue', 'orange', 'green', 'purple', 'red',
                                                                          'blueviolet'])
    plt.ylabel('Naive Bayes Accuracy Score')
    plt.xlabel("Size of Dataset")
    plt.ylim((0.9, 1))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
