
# coding: utf-8

# # 使用RNN进行情绪分析

# 在这里我们将使用RNN(循环神经网络)进行情感分析，至于为什么使用RNN而不是普通的前馈神经网络，是因为RNN能够存储序列单词信息，得到的结果更为准确。这里我们将使用一个带有标签的影评数据集进行训练模型。

# 使用的RNN模型架构如下：
# <img src="assets/network_diagram.png" width=400px>

# 在这里，我们将单词传入到嵌入层而不是使用ONE-HOT编码，是因为词嵌入是一种对单词数据更好的表示。

# 在嵌入层之后，新的表示将会进入LSTM细胞层。最后使用一个全连接层作为输出层。我们使用sigmiod作为激活函数，因为我们的结果只有positive和negative两个表示情感的结果。输出层将是一个使用sigmoid作为激活函数的单一的单元。

# In[5]:

import numpy as np
import tensorflow as tf


# In[6]:

with open('../sentiment-network/reviews.txt', 'r') as f:
    reviews = f.read()
with open('../sentiment-network/labels.txt', 'r') as f:
    labels = f.read()


# In[7]:

print(len(reviews))
print(len(labels))


# In[8]:

reviews[:2000]


# ## 数据预处理

# 构建神经网络的第一步是将数据处理成合适的格式，由于我们需要将数据输入到嵌入层，因此需要将每一个单词
# 编码为整数形式。

# 在数据集中，每条评论是用换行符分隔的。为了解决这些问题，我将把文本分成每一个评论，使用\n作为分隔符。然后我可以把所有的评论组合成一个大的字符串。

# 首先，我们将移除数据中所有的标点符号，然后去掉所有的换行符，得到所有单独的单词组成的数据

# In[9]:

from string import punctuation
all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')

all_text = ''.join(reviews)
words = all_text.split()


# In[10]:

all_text[:2000]


# In[7]:

#words[:100]


# ### 对单词进行编码

# 嵌入查找要求传入整数到网络中，最简单的方法是创建一个从单词到整数的映射的字典。然后我们能将每条评论转换为整数传入网络。

# In[11]:

print(len(words))
print(len(set(words)))
print(len(reviews))


# In[12]:

set_words = set(words)
print(len(set_words))
list_words = list(set_words)
print(len(list_words))


# In[13]:

from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word : ii for ii, word in enumerate(vocab, 1)}
reviews_ints = []

for review in reviews:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])


# In[14]:

print(len(reviews_ints))
print(reviews_ints[1])


# ### 对标签进行编码
# 我们的标签有'positive'和'negative'两种，为了在网络中使用它们，我们需要将两个标签转换为1和0.

# In[15]:

labels = np.array([0 if label == 'negative' else 0 for label in labels.split('\n')])


# In[16]:

review_lens = Counter([len(x) for x in reviews_ints])
print('Zero-length reviews: {}'.format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# 在上面我们发现有一条评论的长度为0，另一方面，有的评论长度太长，对于RNN训练来说，需要太多的步骤，因此我们的处理方法是将每条评论的单词数控制为200个单词。这意味着对于不足200个单词的评论，将用0补上，对于超过200个单词的评论，我们只截取前200个使用。

# In[17]:

#移除长度为0的评论
print(len(reviews_ints))

print(len(reviews_ints[25000]))

revice_len_zero = 0

for i, review in enumerate(reviews_ints,0):
    if len(review) == 0:
        revice_len_zero = i
print(revice_len_zero)

reviews_ints = [review_int for review_int in reviews_ints if len(review_int) > 0]
print(len(reviews_ints))


# In[18]:

print(labels.shape)
labels = labels[:-1]
print(len(labels))


# 现在，我们需要创建一个用于存储输入网络的数据的矩阵。数据来源于reviews_ints，因为我们需要传入数字到网络中，并且每行代表一条评论，长度都是200，对于长度短于200的评论，使用0填充，例如，这是其中一条评论['best', 'movie', ever'],对应的编码是[11,23,354],处理后的行应该是这样：[0,0,0......,11,23,354].对于长度大于200的评论，使用前200个单词作为特征向量。

# In[19]:

seq_len = 200
#处理多余200个单词的评论
reviews_ints = [review[:200] for review in reviews_ints]
#处理少于200个单词的评论
features = []
for review in reviews_ints:
    if len(review) < seq_len : 
        s = []
        for i in range(seq_len - len(review)):
            s.append(0)
        s.extend(review)
        features.append(s)
    else:
        features.append(review)
features = np.array(features)


# In[20]:

features[:10,:100]


# ## 训练、验证、测试数据集划分

# 当把数据处理为网络所需要的shape后，就需要将数据集划分为训练集、验证集、测试数据集

# 在这里，我们定义一个划分系数，split_frac，代表数据保留到训练集中的比例，通常设置为0.8或0.9，然后剩余的数据评分为验证集和测试集。

# In[21]:

split_frac = 0.8

from sklearn.model_selection import train_test_split

train_x, val_x = train_test_split(features, test_size = 1 - split_frac, random_state = 0)
train_y, val_y = train_test_split(labels, test_size = 1 - split_frac, random_state = 0)

val_x, test_x = train_test_split(val_x, test_size = 0.5, random_state = 0)
val_y, test_y = train_test_split(val_y, test_size = 0.5, random_state = 0)

print("\t\tFeatures Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
    "\nValidation set: \t{}".format(val_x.shape),
    "\nTest set: \t\t{}".format(test_x.shape))


# In[22]:

print(len(labels))


# ## 构建图

# 数据预处理完成之后，我们将构建图。第一步是定义好超参数：</br>
# - lstm_size：LSTM细胞隐藏单元数量，稍微设置大点会有不错的效果，常见的值如128, 256, 512等。
# - lstm_layers：网络中LSTM层的数量，这里从1开始，如果不合适就再增加。
# - batch_size：在一次训练中进入网络的数据量。通常情况下，应该设置大一些，如果你能确保内存足够的话。
# - learning_rate：学习率

# In[23]:

lstm_size = 256
lstm_layers = 1
batch_size = 500
learning_rate = 0.1


# In[24]:

print(len(vocab_to_int))


# 对于网络来说，它的输入是200个单词长度的组成的评论向量，每次batch的大小是预设的batch_size个向量。我们会在LSTM层添加dropout，因此会为每个单元被保留的概率提供占位。

# In[25]:

n_words = len(vocab_to_int) + 1
#加1是因为字典从1开始，我们用0来填充

#创建图对象
graph = tf.Graph()

#像图中添加节点
with graph.as_default():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'inputs')
    labels = tf.placeholder(tf.int32, [None, None], name = 'labels')
    keep_prod = tf.placeholder(tf.float32, name = 'keep_prod')


# ### 词嵌入

# 现在我们来添加一个嵌入层。需要这样做的原因是：在我们的词典里有74000个单词，如果使用One-Hot编码来处理将会是非常低效的。为了代替one-hot，我们使用一个嵌入层来作为一个查找表，我们可以使用一个word2vec训练的嵌入层模型，然后在这里加载使用。不过新建一个图并让网络学习权重也是可以的。

# 下面的代码中使用tf.Variable来创建一个嵌入查找矩形，并使用它来使嵌入的向量通过tf.nn.embedding_lookup嵌入查找传递到LSTM单元。这个函数需要两个参数：嵌入矩阵和输入张量，比如一个评论向量。然后它会返回一个带有内嵌向量的张量。因此，如果嵌入层有200个单元，这个函数返回的大小为batch_size, 200]。

# In[26]:

#嵌入向量的大小(嵌入层单元个数)
embed_size = 300

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)


# ## LSTM细胞层

# <img src="assets/network_diagram.png" width=400px>
# </br>
# 接下来我们将创建LSTM层用来构建RNN网络。需要注意的是这里并不是真正的构建图，而仅仅是
# 定义好我们在图中需要的cell的类型。
# </br>
# 我们将使用```tf.contrib.rnn.BasicLSTMCell```来在图中创建LSTM细胞层，，
# 该方法的说明文档如下：
# 
# ```
# tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0, input_size = None,
# state_is_tuple = True, activation = <function tanh at 0x109flef28>)
# ```
# </br>
# 其中，num_units是细胞中单元数量，也就是lstm_size。因此，可以写成
# </br>
# ```lstm = tf.contrib.rnn.BasicLSTMCell(num_units)```
# </br>
# 然后，可以使用```tf.contrib.rnn.DropoutWrapper```来添加dropout。像这样子：
# 
# ```
# drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prod = keep_prod)
# 
# ```
# 
# 大多数情况下，越多的层数会使网络效果更好。这便是深度学习的神奇之处，添加更多的网络层能使得网络可以学习到更多复杂的东西。< /br>
# 此外，还有一个用来创建多层LSTM单元的方式：```tf.contrib.rnn.MultiRNNCell```：
# 
# ```
# cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
# 
# ```
# 解释：```[drop] *```创建了一个长度为lstm_layers的cell列表(drop) ，MultiRNNCell包装器将其构建到RNN的多个层中，其中每个cell为列表中的每个cell。
# 所以你在网络中实际使用的cell其实是有着dropout的多个(或者只有一个)LSTM cell。但是
# 但从体系结构的角度来看，这一切都是一样的，只是单元格中的一个更复杂的图形。

# 在下面的代码中，我们将使用tf.contrib.rnn.BasicLSTMCell去创建LSTM层。然后使用tf.contrib.rnn.DropoutWrapper添加dropout。最后使用tf.contrib.MultiRNNCell创建多个LSTM层。
# 

# In[27]:

with graph.as_default():
    
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prod)
    
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    
    initial_state = cell.zero_state(batch_size, tf.float32)


# ### RNN前向传播

# <img src="assets/network_diagram.png" width=400px>
# 现在我们需要将数据流入RNN节点中，可以使用```tf.nn.dynamic_rnn```来完成。
# 我们需要传入前面创建的RNN(或者多层的LSTM cell以及网络的输入)
# ```
# outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, 
# initial_state=initial_state)
# 
# ```
# 我们创建了一个初始状态initail_state来传入RNN。这是在连续时间步骤中在隐藏层之间传递的cell状态。```tf.nn.dynamic_rnn```做了大部分事情。我们传入cell以及细胞输入，它会处理额外的工作，然后返回每个时间步骤的输出以及最终状态。
# 

# In[28]:

with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)


# ### 输出

# 我们之只关心最终的输出结果，并用来作为情绪预测结果。我们用```outputs[:, -1]来获取最后的输出，并计算与labels的损失

# In[29]:

with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels, predictions)
    
    optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)


# ### 验证准确性

# In[30]:

with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# ### 数据bacth

# In[31]:

def get_batchs(x, y, batch_size = 100):
    n_batchs = len(x) // batch_size
    x, y = x[:n_batchs * batch_size], y[:n_batchs * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# ## 训练

# In[ ]:

epochs = 50

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batchs(train_x, train_y, batch_size), 1):
            feed = {inputs : x,
                   labels : y[:, None],
                   keep_prod : 0.5,
                   initial_state : state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict = feed)
            
        if iteration % 5 == 0:
            print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))
        if iteration % 25 == 0:
            val_acc = []
            val_state = sess.run(cell.zero_state(batch_size, tf.float32))
            for x, y in get_batchs(val_x, val_y, batch_size):
                feed = {
                    inputs : x,
                    labels : y[:, None],
                    keep_prod : 1,
                    initial_state : val_state
                }
                batch_acc, val_state = sess.run([accuracy, final_state], feed_dict = feed)
                val_acc.append(batch_acc)
            print("Val acc: {:.3f}".format(np.mean(val_acc)))
        iteration += 1
    saver.save(sess, 'checkpoints/sentiment.ckpt')


# ## 测试

# In[32]:

test_acc = []
with tf.Session(graph = graph) as sess:
    
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batchs(test_x, test_y, batch_size), 1):
        feed = {inputs : x,
               labels : y[:, None],
               keep_prod : 1,
               initial_state : test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict = feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


# In[ ]:



