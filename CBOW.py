#CBOW(Continuous Bag-of-Word)挑一个要预测的词，来学习这个词前后文中词语和预测词的关系。
#我们这里用到keras生成embedding ,w,b 然后用nce计算loss
from tensorflow import keras
import tensorflow as tf
from utils import process_w2v_data
from visual import show_w2v_word_embedding
#给定输入词（上下文词） 得出相应的词向量
#将词向量输入到神经网络 尝试预测中心词
#比较预测和真实的中心词，计算损失
#利用损失函数和随机优化器来优化神经网络和词嵌入层
corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]
class CBOW(keras.Model):
    def __init__(self,v_dim,emb_dim):
        super().__init__()
        self.v_dim = v_dim
        #词嵌入层：找到所有词的词向量
        self.embeddings = keras.layers.Embedding(
            input_dim = v_dim,output_dim = emb_dim, #[voc_size,embeddng size]
            #embeddng size可以自己定义,越大性能越好 voc_size语料库中的唯一词数
            embeddings_initializer = keras.initializers.RandomNormal(0.,0.1),
        )
        # noise-contrastive estimation 噪声对比估计
        self.nce_w = self.add_weight(
        name = "nce_w",shape=[v_dim,emb_dim],
        initializer = keras.initializers.TruncatedNormal(0.,0.1)
    )
        self.nce_b = self.add_weight(
            name="nce_b",shape=(v_dim,),
            initializer=keras.initializers.Constant(0.1)
        )
        self.opt = keras.optimizers.Adam(0.01)
    #将上下文词向量sum成一个
    # 前向预测部分,把预测时的embedding词向量给拿出来， 然后求一个词向量平均
    def call(self,x,training=None,mask=None):
        # x.shape = [n, skip_window*2]
        o = self.embeddings(x)  # [n, skip_window*2, emb_dim]
        o = tf.reduce_mean(o, axis=1)  # [n, emb_dim]
        return o

    # negative sampling: take one positive label and num_sampled negative labels to compute the loss
    # in order to reduce the computation of full softmax
    #有效损失的近似
    def loss(self,x,y,training=None):
        embedded = self.call(x,training)
        return tf.reduce_mean(
            # nce_loss,它不关心所有词汇loss， 而是抽样选取几个词汇用来传递loss
            tf.nn.nce_loss(
                weights = self.nce_w,biases=self.nce_b,labels=tf.expand_dims(y,axis=1),
                inputs = embedded,num_sampled=5,num_classes=self.v_dim
            )
        )
    def step(self,x,y):
        with tf.GradientTape() as tape:
            loss = self.loss(x,y,True)
            grads = tape.gradient(loss,self.trainable_variables) #计算w的梯度
        self.opt.apply_gradients(zip(grads,self.trainable_variables)) #更新梯度
        return loss.numpy()
def train(model,data):
    for t in range(2500):
        bx,by = data.sample(8)
        loss = model.step(bx,by)
        if t % 200 == 0:
            print("step : {} | loss:{}".format(t,loss))

if __name__ == "__main__":
    #将取出的两边词和中心词 以及v2i、i2v传到Dataset
    d = process_w2v_data(corpus,skip_window=2,method="cbow")
    m=CBOW(d.num_word,2) #d.num_word == len(v2i)
    train(m,d)

    show_w2v_word_embedding(m,d,"./visual/results/cbow.png")
