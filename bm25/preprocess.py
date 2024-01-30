import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from gensim.summarization import bm25



class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def tokenize(self, text):
        return word_tokenize(text)

    def lemmatize(self, token):
        return self.lemmatizer.lemmatize(token)

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stopwords]

    def preprocess(self, text):
        tokens = self.tokenize(text)
        tokens = [self.lemmatize(token.lower()) for token in tokens]
        tokens = self.remove_stopwords(tokens)
        return tokens
    


preprocessor = Preprocessor()
# processed_corpus = []
# with open("corpus.txt","r") as f:
#     for doc in f.readlines():
#         tokens = preprocessor.preprocess(doc)
#         processed_corpus.append(tokens)


with open("bm25model.pkl","rb") as f:
    bm25Model = pickle.load(f)
    f.close()
test_str = preprocessor.preprocess("Augmented reality, deep learning and vision-language query system for construction worker safetyChen")
scores = bm25Model.get_scores(test_str)
print('测试句子：', test_str)
def get_top5_indices(lst):
    indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:5]
    return indices
print(get_top5_indices(scores))
# for i, j in zip(scores, processed_corpus):
#     print('分值：{},原句：{}'.format(i, j))
# print('\n')


