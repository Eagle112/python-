from __future__ import division
import nltk,re,pprint
from bs4 import BeautifulSoup
from nltk import word_tokenize
from urllib import request
url = "https://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode("utf8")
print(type(raw),"\n", len(raw),"\n")
print(raw[:80])
#%% #分段执行
tokens = word_tokenize(raw)#分词
print(tokens[1020:1060])
print(len(tokens))
#%%
text = nltk.Text(tokens)
print(type(text))
print(text[1020:1060])
#%%
# print(text.collocation_list()) #源代码有错 deep3
print(raw.find("PART I"))
print(raw.find("End"))
#%%
raw = raw[5303:1157812]
print(raw)
#%%
from urllib.request import urlopen
from lxml import etree
import nltk
from bs4 import BeautifulSoup
html = BeautifulSoup(open("nltk/BBC NEWS _ Health _ Blondes 'to die out in 200 years'.html", encoding='latin1'), features='lxml')#读取本地文件(deep3)
# htmls = str(html)#对象转字符串
# print(htmls[:60])
raw =html.get_text()
# raw = str(raw)
tokens = nltk.word_tokenize(raw)
tokens = tokens[96:399]
text = nltk.Text(tokens)
print(text.concordance('gene'))
# print(tokens)
# print(text)

#%%  #此段无效
import feedparser
llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
print(llog['feed']['title'])
print(len(llog.entries))
#%%  #打开本地文件
import os
# print(os.listdir('.'))#打印当前目录下的文件名
f = open('nltk/dict.txt','rU')
# print(f.read())
for line in f:
    print(line.strip())
#%%
path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
raw = open(path, 'rU').read()

#%% #捕获用户输入
import nltk
s = input("Enter some text: ")
#%%
print("You typed", len(nltk.word_tokenize(s)), "words.")

#%%  字符串格式
# raw = open('document.txt').read()
# tokens = nltk.word_tokenize(raw)
# words = [w.lower() for w in tokens]
couplet = "Shall I compare thee to a Summer's day?"\
"Thou are more lovely and more temperate:"
print(couplet)

#%%
import codecs
path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
f = open(path, encoding='latin2')
for line in f:
    line = line.strip()
    print(line)
#%%
nacute = u'\u0144'
nacute_utf = nacute.encode('utf8')
print(nacute_utf)
#%%
import unicodedata
path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
lines = open(path, encoding='latin2').readlines()
line = lines[2]
print(line.encode('unicode_escape'))
#%%
import re
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
# print([w for w in wordlist if re.search('ed$', w)])
wsj = sorted(set(nltk.corpus.treebank.words()))
fd = nltk.FreqDist(vs for word in wsj for vs in re.findall(r'[aeiou]{2,}', word))
print(fd.items())
date = [n for n in re.findall('[0-9]{2,4}', '2009-12-31')]#输出时间
print(date)
#%%
import re
import nltk
# regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'#以元音开头的或以元音结尾的,或非元音
# def compress(word):
#     pieces = re.findall(regexp, word)
#     return ''.join(pieces)
# english_udhr = nltk.corpus.udhr.words('English-Latin1')
# print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))
# rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
# cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
# cfd = nltk.ConditionalFreqDist(cvs)
# cfd.tabulate()
# raw = """DENNIS: Listen, strange women lying in ponds distributing swords
# is no basis for a system of government. Supreme executive power derives from
# a mandate from the masses, not from some farcical aquatic ceremony."""
# tokens = nltk.word_tokenize(raw)
# porter = nltk.PorterStemmer()
# lancaster = nltk.LancasterStemmer()
# print([porter.stem(t) for t in tokens])
# class IndexedText(object):
#     def __init__(self, stemmer, text):
#         self._text = text
#         self._stemmer = stemmer
#         self._index = nltk.Index((self._stem(word), i)for (i, word) in enumerate(text))
#     def concordance(self, word, width=40):
#         key = self._stem(word)
#         wc = int(width / 4)
#         for i in self._index[key]:
#             lcontext = ' '.join(self._text[i - wc:i])
#             rcontext = ' '.join(self._text[i:i + wc])
#             ldisplay = '%*s' % (width, lcontext[-width:])
#             rdisplay = '%-*s' % (width, rcontext[:width])
#             print(ldisplay, rdisplay)
#     def _stem(self, word):
#         return self._stemmer.stem(word).lower()
# porter = nltk.PorterStemmer()
# grail = nltk.corpus.webtext.words('grail.txt')
# text = IndexedText(porter, grail)
# print(text.concordance('lie'))
#%%
from nltk.tokenize.regexp import RegexpTokenizer
text = 'That U.S.A. poster-print costs $12.40...'
pattern = r"""(?x)
(?:[A-Z]\.)+
|\d+(?:\.\d+)?%?
|\w+(?:[-']\w+)*
|\.\.\.
|(?:[.,;"'?():-_`]) 
"""
tokeniser=RegexpTokenizer(pattern)
#%%
tokeniser.tokenize(text)
#%% #分割(断句)
import nltk
import pprint
sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = sent_tokenizer.tokenize(text)
pprint.pprint(sents[171:181])
#%%
colors = 'rgbcmyk' # red, green, blue, cyan, magenta, yellow, black
def bar_chart(categories, words, counts):
    "Plot a bar chart showing counts for each word by category"
    import pylab
    ind = pylab.arange(len(words))
    width = 1 / (len(categories) + 1)
    bar_groups = []
    for c in range(len(categories)):
        bars = pylab.bar(ind + c * width, counts[categories[c]], width,
                         color=colors[c % len(colors)])
        bar_groups.append(bars)
    pylab.xticks(ind + width, words)
    pylab.legend([b[0] for b in bar_groups], categories, loc='upper left')
    pylab.ylabel('Frequency')
    pylab.title('Frequency of Six Modal Verbs by Genre')
    matplotlib.use('Agg')
    pylab.savefig('modals.png')
    print('Content-Type: text/html')
    print('<html><body>')
    print('<img src="modals.png"/>')
    print('</body></html>')
genres = ['news', 'religion', 'hobbies', 'government', 'adventure']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfdist = nltk.ConditionalFreqDist((genre, word) for genre in genres for word in nltk.corpus.brown.words(categories=genre) if word in modals)
counts = {}
for genre in genres:
    counts[genre] = [cfdist[genre][word] for word in modals]
bar_chart(genres, modals, counts)
#%%
import networkx as nx
import matplotlib
from nltk.corpus import wordnet as wn
def traverse(graph, start, node):
    graph.depth[node.name] = node.shortest_path_distance(start)
    for child in node.hyponyms():
        graph.add_edge(node.name, child.name)
        traverse(graph, start, child)
def hyponym_graph(start):
    G = nx.Graph()
    G.depth = {}
    traverse(G, start, start)
    return G
def graph_draw(graph):
    nx.draw_graphviz(graph,node_size = [16 * graph.degree(n) for n in graph],node_color = [graph.depth[n] for n in graph],with_labels = False)
    matplotlib.pyplot.show()
dog = wn.synset('dog.n.01')
graph = hyponym_graph(dog)
graph_draw(graph)
#%%
import csv
input_file = open("nltk/lexicon.csv", "rt")
for row in csv.reader(input_file):
    print(row)
#%% 词性标注
text = nltk.word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))
#%%
tagged_token = nltk.tag.str2tuple('fly/NN')
print(tagged_token)
#%%
# nltk.corpus.brown.tagged_words()
from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(tagset='universal')
word_tag_pairs = nltk.bigrams(brown_news_tagged)
print(list(nltk.FreqDist(a[1] for (a, b) in word_tag_pairs if b[1] == 'N')))
#%%
wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd = nltk.FreqDist(wsj)
cfd1 = nltk.ConditionalFreqDist(wsj)
cfd1['yield'].keys()
#%%
brown_learned_text = brown.words(categories='learned')
sorted(set(b for (a, b) in nltk.bigrams(brown_learned_text) if a == 'often'))
brown_lrnd_tagged = brown.tagged_words(categories='learned',tagset='universal')
tags = [b[1] for (a, b) in nltk.bigrams(brown_lrnd_tagged) if a[0] == 'often']
fd = nltk.FreqDist(tags)
fd.tabulate()
    #%%
from nltk.corpus import brown
def process(sentence):
    for (w1, t1), (w2, t2), (w3, t3) in nltk.trigrams(sentence):
        if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
            print(w1, w2, w3)

for tagged_sent in brown.tagged_sents():
    process(tagged_sent)
#%%
import nltk
alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = list(vocab)[:1000]
mapping = nltk.defaultdict(lambda: 'UNK')
for v in v1000:
    mapping[v] = v
alice2 = [mapping[v] for v in mapping]
print(alice2[:100])
#%%
from nltk.corpus import brown
fd = nltk.FreqDist(brown.words(categories='news'))
brown_tagged_sents = brown.tagged_sents(categories='news')
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.keys()[:100]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)
#%%  训练和测试
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
print(unigram_tagger.tag(brown_sents[2007]))
#%%
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
print(unigram_tagger.evaluate(test_sents))

#%%  请注意，bigram 标注器能够标注训练中它看到过的句子中的所有词，但对一个没见过 的句子表现很差。只要遇到一个新词（如13.5），就无法给它分配标记。它不能标注下面的 词（如：million)，即使是在训练过程中看到过的，只是因为在训练过程中从来没有见过它 前面有一个None 标记的词。因此，标注器标注句子的其余部分也失败了。它的整体准确度 得分非常低：
bigram_tagger = nltk.BigramTagger(train_sents)
print(bigram_tagger.tag(brown_sents[2007]))
unseen_sent = brown_sents[4203]
print(bigram_tagger.tag(unseen_sent))
#%%
import _pickle as cPickle

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
output = open('t2.pkl', 'wb')
cPickle.dump(t2, output, -1)
output.close()
#%%
import _pickle as cPickle
from _pickle import load
input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()
#%%
text = """The board's action shows what free enterprise
is up against in our complex maze of regulatory laws ."""
tokens = text.split()
print(tagger.tag(tokens))
#%%
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t2.evaluate(test_sents)




