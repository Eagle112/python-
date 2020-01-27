from nltk.book import *
import nltk
import jieba
def lexical_diversity(text):
    return len(text)/len(set(text))
def percentage(count,total):
    return 100*count/total
# 链表,不是数组
# sent1 = ['call','me','ishmal','.']
# print(text5[16715:])
# fdist1=FreqDist(text1)
# vocabulary1=fdist1.most_common(50)
#print(vocabulary1)#打印前50个频率最高的词
#print(fdist1['whale'])#whale出现的频率
# fdist1.plot(50,cumulative=True)#前50个单词的累积图
# print(fdist1.hapaxes())#统计低频词
# V=set(text1)
# long_words = [w1 for w1 in V if len(w1)==15]#长度为15的词
# print(sorted((long_words)))
# print(bigrams(['more','is','said','than','done']))
#print(text4.collocation_list())#频率高的双连词
# fdist = FreqDist([len(w) for w in text1])
# print(fdist.items())
# print(fdist.max())
# print(fdist[3])
# print(fdist.freq(3))#频率
# fdist.tabulate()#绘制频率分布表
# --------------------------------------------------------------
# print(sent7)#sent7是一个句子
# print([w for w in sent7 if len(w)<4])
# print(sorted([w for w in set(text1) if w.endswith('ableness')])) #词尾是ableness的单词
# print(sorted([w for w in set(text7) if '-' in w and 'index' in w]))#测试几个语法
# print([w.upper() for w in set(text1) if len(w)==5])#给每个单词大小写
# print(len(set([w.upper() for w in text1])))#转化为大写后再统计数量,可以排除因为大小写不同而认为是不同的单词
# babelize_shell()已经不使用
# nltk.chat.chatbots()#聊天机器人
# -------------------------------------------------------------------------------
from nltk.corpus import gutenberg
# emma=nltk.Text(gutenberg.words('austen-emma.txt'))#古藤堡预料库
# print(len(emma))
# print(emma.concordance("surprize"))
# print(gutenberg.raw('austen-emma.txt'))#有多少个字母
from nltk.corpus import webtext
# for fileid in webtext.fileids():
#     print(fileid,webtext.raw(fileid)[0:60])
from nltk.corpus import nps_chat
# chatroom = nps_chat.posts('10-19-20s_706posts.xml')
# print(chatroom)
#布朗语料库
from nltk.corpus import  brown
# print(brown.categories())
# print(brown.words(categories='news'))
# ------------------布朗语料库---------------------------------
from nltk.corpus import reuters
# print(reuters.fileids())
# print(reuters.categories())
# print(reuters.categories('training/9865'))#查看文章分类
# print(reuters.fileids('barley'))#查看包含这个分类的文章
# print(reuters.words(['training/9865','training/9880'])[0:100])#显示文章中的单词
from nltk.corpus import inaugural
# print(inaugural.fileids())
# print([fileid[0:4] for fileid in inaugural.fileids()])
# cfd = nltk.ConditionalFreqDist((target, fileid[:4]) for fileid in inaugural.fileids() for w in inaugural.words(fileid) for target in ['america','citizen'] if w.lower().startswith(target))
# cfd.plot()#词汇 america 和 citizen 随时间推移的使用情况
# -------------------------------------------------------------------------------------
# print(nltk.corpus.cess_esp.words())
# print(nltk.corpus.floresta.words())
# print(nltk.corpus.indian.words('hindi.pos'))
# print(nltk.corpus.udhr.fileids())#udhr是世界人权宣言
# print(nltk.corpus.udhr.words('Javanese-Latin1')[11:])
from nltk.corpus import udhr
# languages = ['Chickasaw', 'English', 'German_Deutsch','Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
# cfd = nltk.ConditionalFreqDist((lang, len(word)) for lang in languages for word in udhr.words(lang + '-Latin1'))
# cfd.plot(cumulative=False)
# raw_text=udhr.raw(fileids='Lingala-Latin1')#指定语言的频率图
# nltk.FreqDist(raw_text).plot()
# raw = gutenberg.raw("burgess-busterbrown.txt")
# print(raw[1:20]) #raw代表的字母
from nltk.corpus import PlaintextCorpusReader
# corpus_root = 'D:/pythonproject/nltk'#导入自己的语料库
# wordlists = PlaintextCorpusReader(corpus_root, '.*')
# print(wordlists.fileids())
# print(wordlists.words('dict.txt'))
genre_word = [(genre,word) for genre in ['news','romance'] for word in brown.words(categories=genre)]#条件频率分布，类似双重嵌套循环
# print(len(genre_word))
# print(genre_word[:4])
# print(genre_word[-4:])
cfd = nltk.ConditionalFreqDist(genre_word)
# print(cfd)
# print(cfd['news'])
# print(list(cfd['romance']))
# print(cfd['romance']['could'])
from nltk.corpus import inaugural
# cfd = nltk.ConditionalFreqDist((target,fileid[:4]) for fileid in inaugural.fileids() for w in inaugural.words(fileid) for target in ['america','citizen'] if w.lower().startswith(target))#括号中第一个参数是线条,第2个参数是横坐标
# cfd.plot()
from nltk.corpus import udhr
# languages = ['Chickasaw', 'English', 'German_Deutsch','Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
# cfd = nltk.ConditionalFreqDist( (lang, len(word)) for lang in languages for word in udhr.words(lang + '-Latin1'))
# cfd.tabulate(conditions=['English', 'German_Deutsch'], samples=range(10), cumulative=True)
# ------------------------------------------------------------------------------------------
def generate_model(cfdist, word, num=15):#文本生成
    for i in range(num):
        print(word),
        word = cfdist[word].max()
text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)
print(generate_model(cfd, 'living'))