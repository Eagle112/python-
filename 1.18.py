import nltk
from nltk.corpus import stopwords

def unusual_words(text):
    text_vocab=set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)

# print(unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt')))#打印错别字或罕见字
# entries = nltk.corpus.cmudict.entries()
# for entry in entries[39943:39951]:
#     print(entry)
# prondict = nltk.corpus.cmudict.dict()
# prondict['blog'] = [['B', 'L', 'AA1', 'G']]#CMU字典里有发音
# text = ['natural', 'language', 'processing']
# print([ph for w in text for ph in prondict[w]])
# print(prondict['blog'])
from nltk.corpus import swadesh
# print(swadesh.words('en'))#打印英语中的常用词
fr2en = swadesh.entries(['fr', 'en'])#打印两种语言的同源词
# print(fr2en)
# translate = dict(fr2en)#将同源词转化为字典
# print(translate['chien'])
# de2en = swadesh.entries(['de', 'en'])
# es2en = swadesh.entries(['es', 'en'])
# translate.update(dict(de2en))#为词典添加新的语言
# translate.update(dict(es2en))
# print(translate['Hund'])
from nltk.corpus import toolbox
# print(toolbox.entries('rotokas.dic'))
# --------------------------------------------------------------------------------------------------------------
from nltk.corpus import wordnet as wn
# print(wn.synsets('motorcar'))#打印同义词
# print(wn.synset('car.n.01').lemma_names())#同义词集合
# print(wn.synset('car.n.01').definition())#car的定义
# print(wn.synset('car.n.01').examples())#car的例句
# print(wn.synset('car.n.01').lemmas())#同义词集合
# for synset in wn.synsets('car'):
#     print(synset.lemma_names())
# print(wn.synset('smasher.n.02').lemma_names())
motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()#下位词
# print(types_of_motorcar[26])
# print(sorted([lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas()]))
# print(motorcar.hypernyms())#上位词
# paths = motorcar.hypernym_paths()
# print(len(paths))
# print([synset.name() for synset in paths[1]])
# print(motorcar.root_hypernyms())
# nltk.app.wordnet()#打开wordnet
# print(wn.synset('tree.n.01').part_meronyms())#更多词汇关系
# print(wn.synset('tree.n.01').substance_meronyms())
# print(wn.synset('tree.n.01').member_holonyms())
# for synset in wn.synsets('mint', wn.NOUN):
#     print(synset.name() + ':', synset.definition())
# print(wn.synset('walk.v.01').entailments())#走路的动作包括抬脚的动作
# print(wn.synset('eat.v.01').entailments())#吃的动作包含咀嚼和吞咽
# print(wn.lemma('supply.n.02.supply').antonyms())#反义词
# print(dir(wn.synset("eat.v.01")))#查看synest有哪些方法
right = wn.synset('right_whale.n.01')
orca = wn.synset('orca.n.01')
minke = wn.synset('minke_whale.n.01')
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')
# print(right.lowest_common_hypernyms(minke))#寻找关系最近的同义词集
# print(right.lowest_common_hypernyms(orca))#
# print(right.lowest_common_hypernyms(tortoise))
# print(wn.synset('baleen_whale.n.01').min_depth())#查看在wordnet中的层级,层级越大表示这个词越具体
# print(right.path_similarity(minke))#输出一个小数,得出两者的相似度
# help(wn)
help(help)








