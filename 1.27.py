def gender_features(word):
    return {'suffix1': word[-1],'suffix2':word[-2]}
print(gender_features('Shrek'))
#%%
from nltk.corpus import names
import nltk
import random
names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(names)
from nltk.classify import apply_features
featuresets = [(gender_features(n), g) for (n, g) in names]
train_set = apply_features(gender_features, names[500:])
test_set = apply_features(gender_features, names[:500])
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(classifier.classify(gender_features('Trinity')))#测试结果
print(nltk.classify.accuracy(classifier, test_set))
#%%
def gender_features2(name): #训练器
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    return features
gender_features2('John')

featuresets = [(gender_features2(n), g) for (n, g) in names] #特征集
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)#分类器

print(nltk.classify.accuracy(classifier, test_set))
#%%
train_names = names[1500:]
devtest_names = names[500:1500]
test_names = names[:500]
train_set = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print((nltk.classify.accuracy(classifier, devtest_set)))
#%%
errors = [] #预测错误集合
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append((tag, guess, name))

for (tag, guess, name) in sorted(errors): # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
print('correct=%-8s guess=%-8s name=%-30s' %(tag, guess, name))
#%%
from nltk.corpus import movie_reviews
import random
import nltk
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words.keys())[:2000]#勘误
print(word_features)
#%%
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
print(document_features(movie_reviews.words('pos/cv957_8737.txt')))
#%%
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
#%%
from nltk.corpus import brown
suffix_fdist = nltk.FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] +=1 #P212勘误
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-2:]] += 1
common_suffixes = list(suffix_fdist.keys())[:100]
print(common_suffixes)






