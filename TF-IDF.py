#2021.4.23  代码参考自莫烦 https://github.com/MorvanZhou/NLP-Tutorials
#搜索引擎(TF-IDF检索) 通过这个例子让我们熟悉nlp中重要的部分就是把语言向量化
#我们搜索文档时会搜索关键信息，通过关键信息的检索返回结果，TF-IDF就是做信息提取的工作
#TF：Term Frequency  词频
#IDF:inverse Document Frequency 突出具有代表性的关键词
import numpy as np
from collections import Counter
import itertools
#计算与搜索句子匹配度的文档 相当于搜索引擎数据库中的文档
docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]
#对每个文档分词，docs_words是个二维列表：类似[['I', 'love', 'you'], ['I', 'miss', 'you']]
docs_words=[d.replace(",","").split(" ") for d in docs]
#统计单词，用集合过滤掉重复的 vocab类似：{'miss', 'you', 'I', 'love'}
vocab = set(itertools.chain(*docs_words))
#itertools.chain可迭代取出每个元素 之后我们可以把它转化成我们想要的类型
v2i={v:i for i,v in enumerate(vocab)}
i2v={i:v for v,i in v2i.items()}
#v2i结果类似{'miss': 0, 'you': 1, 'I': 2, 'love': 3}
#i2v结果类似{0: 'miss', 1: 'you', 2: 'I', 3: 'love'}
def safe_log(x):
    mask = x!=0
    x[mask] = np.log(x[mask])
    return x
#计算tf的方法
tf_methods={
    #lambda可将函数作为参数传进去 x是参数 返回:后的内容
    "log":lambda x:np.log(1+x),
    "augmented":lambda x:0.5+0.5*x/np.max(x,axis=1,keepdims=True),
    "boolean":lambda x:np.minimum(x,1),
    "log_avg":lambda x:(1+safe_log(x))/(1+safe_log(np.mean(x,axis=1,keepdims=True))),
}
#计算idf的方法
idf_methods = {
    "log":lambda x:1+np.log(len(docs)/(x+1)),
    "prob":lambda x:np.minimum(0,np.log((len(docs)-x)/(x+1))),
    "len_norm":lambda x:x/(np.sum(np.square(x))+1),
}
#文档中词出现的总数
def get_tf(method="log"):
    #term frequency:how frequent a word appears in a doc
    _tf = np.zeros((len(vocab),len(docs)),dtype=np.float64) #初始化一个矩阵 每列代表一个文档
    for i,d in enumerate(docs_words):
        counter = Counter(d) #统计文档中每个词在该文档中出现的次数
        for v in counter.keys():  #counter 是一个字典
            _tf[v2i[v],i]=counter[v]/counter.most_common(1)[0][1]
            #每个词出现的次数/这个文档某单词出现的最多次数==相对频率 放到第i列单词相对的位置
    weighted_tf = tf_methods.get(method,None)
    if weighted_tf is None: #如果方法不存在 报错
        raise ValueError
    return weighted_tf(_tf)

#得到每个词的权重向量
def get_idf(method="log"):
    df = np.zeros((len(i2v),1))
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_words: #统计每个词在所有文档中出现的次数 以便分配权重
            d_count+= 1 if i2v[i] in d else 0
        df[i,0]=d_count

    idf_fn=idf_methods.get(method,None)
    if idf_fn is None:
        raise ValueError
    return idf_fn(df)

#计算cos距离
def cosine_similarity(q,_tf_idf):
    unit_q = q / np.sqrt(np.sum(np.square(q),axis=0,keepdims=True))
    unit_ds = _tf_idf/np.sqrt(np.sum(np.square(_tf_idf),axis=0,keepdims=True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity

def docs_score(q,len_norm=False):
    q_words = q.replace(",","").split(" ") #分词

    #add unknown words
    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:  #如果句子中的词不在文档的词表中，就添加到词表
            v2i[v]=len(v2i)
            i2v[len(v2i)-1]=v
            unknown_v+=1 #有unknown_v 个未在词表中的词
    if unknown_v>0:
        #将新词的idf=0 和词表的idf拼接
        _idf=np.concatenate((idf,np.zeros((unknown_v,1),dtype=np.float)),axis=0)
        _tf_idf = np.concatenate((tf_idf,np.zeros((unknown_v,tf_idf.shape[1]),dtype=np.float)),axis=0)
    else:
        _idf,_tf_idf = idf,tf_idf
    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf),1),dtype=np.float)
    for v in counter.keys():   #计算句子的词频
        q_tf[v2i[v],0]=counter[v]

    q_vec = q_tf*_idf #将句子转化成向量
    q_scores = cosine_similarity(q_vec,_tf_idf) #计算句子和文档的cos
    if len_norm:
        len_docs = [len(d) for d in docs_words]
        q_scores=q_scores/np.array(len_docs)
    return q_scores

#打印前三个文档每个文档中TF-IDF值最高的两个单词
def get_keywords(n=2):
    for c in range(3):
        col = tf_idf[:,c]
        idx = np.argsort(col)[-n:] #返回从小到大排序的元素的索引
        print("doc{},top{} keywords {}".format(c,n,[i2v[i] for i in idx]))

tf = get_tf()
idf=get_idf()
tf_idf=tf*idf
print("tf shape(vecb in each docs):",tf.shape)
print("\ntf samples:\n",tf[:2])
print("\nidf shape(vecb in all docs):",idf.shape)
print("\nidf samples:\n",idf[:2])
print("\ntf_idf shape:",tf_idf.shape)
print("\ntf_idf sample:\n",tf_idf[:2])

#test code
get_keywords()
q = "I get a coffee cup"


scores = docs_score(q)
print(scores)
d_ids = scores.argsort()[-3:][::-1] #从后向前取3个TF-IDF值最大的索引

print("\ntop 3 docs for '{}':\n{}".format(q,[docs[i] for i in d_ids]))




