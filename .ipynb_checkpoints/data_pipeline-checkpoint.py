import re  # Regular Expressions
import pickle # To serialize Python objects
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # Vectoriers
from gensim.models import Word2Vec  # Word2Vec Model
import nltk  # Natural Language Toolkit
nltk.download('stopwords')  # Download the stopwords corpus
nltk.download('wordnet')  # Download the WordNet corpus

nltk.download('punkt')  # Download the Punkt tokenizer
import scipy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions # Contractions
# from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# define the data pipeline class
class ETL_Pipeline():
    '''This class will implement a basic ETL pipeline for text data.'''
    def __init__(self) -> None:
        self.data_path = None
        self.data = None
        self.text = None
        self.preprocessed_text = None

        self.tokenizer = None
        self.tokens = None
        self.lemmatizer = None
        self.vocabulary = None
    
    def extract(self, data_path):
        '''This function should take in a file path and read the file. 
        The function should return a pandas DataFrame containing the full dataset.
        The data should be stored in the self.data attribute. 
        The text data should be stored in the self.text attribute.'''
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        print('Data loaded successfully!')
        # Coerce the text data to string type
        self.data['text'] = self.data['text'].astype(str)
        self.text = self.data['text']  # Extract the text data from the DataFrame
        print('Text data extracted successfully!')
    def preprocess(self):
        '''
        Input: This function takes in a pandas.Series() of a corpus of text data as an argument.
        Output: This function should output an indexed vocabulary and preprocessed tokens.

        Preprocessing steps include:
        1. Lowercasing the text
        2. Removal of HTML tags
        3. Removal of special characters
        4. Expansion of contractions
        5. Removal of stopwords
        6. Tokenization
        7. Lemmatization
        8. Create a vocabulary
        '''
        # Lowercasing the text
        self.text = self.lowercase(self.text)
        # Removal of HTML tags
        # self.text = self.remove_html_tags(self.text)
        # Expansion of contractions
        self.text = self.text.apply(self.expand_contractions)
        # Removal of special characters
        self.text = self.text.apply(self.remove_special_characters)
        # Removal of stopwords
        self.text = self.text.apply(self.remove_stopwords)
        # Store the preprocessed text data in a new attribute
        self.preprocessed_text = self.text
        print('Text data preprocessed successfully!')

        # Tokenization: splits the text into individual words
        self.tokens = self.text.apply(self.tokenize)
        # Lemmatization: transforms the word into their semantically correct base forms
        self.tokens = self.tokens.apply(self.lemmatize)
        print('Text data tokenized and lemmatized successfully!')

        # Create a vocabulary
        self.vocabulary = self.tokens.apply(self.create_vocabulary)
        print('Vocabulary created successfully!')
        return self.vocabulary, self.tokens

    def encode(self, method='word2vec'):
        '''
        Input: This function operates on the preprocessed token outputs of the 'preprocess()' function. This function also takes in a method argument that specifies the type of encoding to use. 
        The default method is 'bow' (bag of words). Other options are 'td-idf' and 'word2vec'.
        
        Output: This function should output a vectorized representation of the text data. The vectorized data is also saved as a serialized object in the 'data' directory.
        '''
        if method == 'bow':
            # Bag of Words Encoding
            vectorizer = CountVectorizer()
            vectorized_data = vectorizer.fit_transform(self.tokens)

            with open('./data/bow_vectorized_data.pkl', 'wb') as f:
                pickle.dump(vectorized_data, f)
            print('Bag of Words Encoding completed successfully!')            
            return vectorized_data
        elif method == 'td-idf':
            # TF-IDF Encoding
            vectorizer = TfidfVectorizer()
            vectorized_data = vectorizer.fit_transform(self.tokens)

            with open('./data/tfidf_vectorized_data.pkl', 'wb') as f:
                pickle.dump(vectorized_data, f)
            print('TF-IDF Encoding completed successfully!')
            return vectorized_data
        elif method == 'word2vec':
            # Word2Vec Encoding
            model = Word2Vec(sentences=self.tokens, vector_size=100, window=5, min_count=1, workers=4)
            model.build_vocab(self.tokens)
            model.train(self.tokens, total_examples=model.corpus_count, epochs=10)
            vectorized_data = []
            for token in self.tokens:
                if token in model.wv:
                    vectorized_data.append(model.wv[token])
                else:
                    print(f'Word {token} not in vocabulary!')
            # vectorized_data = model.wv[self.vocabulary]
            with open('./data/word2vec_vectorized_data.pkl', 'wb') as f:
                pickle.dump(vectorized_data, f)
            print('Word2Vec Encoding completed successfully!')
            return vectorized_data
        else:
            raise AttributeError("Invalid method argument. Please specify a valid encoding method: 'bow', 'td-idf', or 'word2vec'.")

    def get_sentiment(self):
        '''This function should return the sentiment labels for the data based on the 'rating' column of the data.
        The sentiment labels should be binary, with ratings >= 3 being positive and ratings < 3 being negative.
        The positive sentiment should be encoded as 1 and the negative sentiment should be encoded as 0.'''
        self.data['sentiment'] = self.data['rating'].apply(lambda x: 1 if x >= 3 else 0)
        print('Sentiment labels extracted successfully!')
    
    def transform(self):
        '''This function should preprocess the text data using the 'preprocess()' function,
        and then transform the text data into vectorized format using the 'encode()' function.
        The function should also encode the sentiment labels using the 'get_sentiment()' function.'''
        
        # Remove rows with missing text or rating values
        self.data = self.data.dropna(subset=['text', 'rating'])

        # Remove unnecessary features from the data
        self.data = self.data[['text', 'rating']]

        # Get the sentiment labels
        self.get_sentiment()

        # Remove HTML tags from the text data
        self.data['text'] = self.data['text'].apply(self.remove_html_tags)
        self.text = self.data['text']

        # Preprocess the text data
        print('Preprocessing the text data...')
        self.vocabulary, self.tokens = self.preprocess()
        self.data['tokens'] = self.tokens
        self.data['text'] = self.preprocessed_text  # Replace the text data with the preprocessed text
        print(self.data.head())

        # Remove rows with empty token lists
        empty_tokens = self.data[self.data['tokens'].apply(len) == 0].index
        if len(empty_tokens) > 0:
            print(f'Removing {len(empty_tokens)} rows with empty token lists...')
            self.data = self.data.drop(empty_tokens)

        # Encode the text data
        print('Encoding the text data...')
        self.encode(method='word2vec') # Use Word2Vec encoding
        print('Data transformed successfully!')

    def load(self, filename):
        '''This function saves the transformed data as a serialized object in the data directory.'''
        print('Saving transformed data...')
        if self.data is not None:
            with open(f'./data/{filename}.pkl', 'wb') as f:
                pickle.dump(self.data, f)
            print('Transformed data saved successfully!')
        else:
            raise FileNotFoundError('No transformed data found. Please run the extract() and transform() functions first.')
    ##### Preprocessing Functions #####
    def lowercase(self, text):
        '''This function converts text to lowercase.'''
        return text.str.lower()
    def remove_special_characters(self, text):
        '''This function removes special characters from a string.'''
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        return text
    def remove_html_tags(self, text):
        '''This function removes HTML tags from a string.'''
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    def remove_stopwords(self, text):
        '''This function removes stopwords from a string.'''
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
        return text
    def expand_contractions(self, text):
        '''This function expands contractions in a string.'''
        expanded_text = contractions.fix(text)
        return expanded_text
    def tokenize(self, text):
        '''This function tokenizes a string.'''
        tokenizer = nltk.tokenize.WhitespaceTokenizer()
        tokens = tokenizer.tokenize(text)
        return tokens
    def lemmatize(self, tokens):
        '''This function lemmatizes words.'''
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
    def create_vocabulary(self, tokens):
        '''This function creates a vocabulary from a list of tokens.'''
        vocabulary = set([word for token in tokens for word in token])
        return vocabulary