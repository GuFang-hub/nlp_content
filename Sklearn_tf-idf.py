#sklearn 解决文章增多单词量增多导致的内存占用大的问题
#并不是每一个文章都提及所有词汇  引入sparse Matrix 稀疏矩阵解决占内存大的问题
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
vectorizer=TfidfVectorizer()
tf_idf=vectorizer.fit_transform(docs) #计算出每个文档的向量表示
#每个词的idf值
print("idf: ",[(n,idf) for idf,n in zip(vectorizer.idf_,vectorizer.get_feature_names())])
print("v2i: ", vectorizer.vocabulary_) #词汇表中去掉了停用词

q="I get a coffee cup"
qtf_idf = vectorizer.transform([q])
res = cosine_similarity(tf_idf,qtf_idf)
res = res.ravel().argsort()[-3:] #ravel() flattened array
print("\ntop 3 docs for '{}':\n{}".format(q,[docs[i] for i in res[::-1]]))



