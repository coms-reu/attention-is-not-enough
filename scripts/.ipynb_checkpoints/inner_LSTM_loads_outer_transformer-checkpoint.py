#!/usr/bin/env python
# coding: utf-8

# # Nested encoder that will take in EncTransformDec

# In[1]:


import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import sys

import numpy as np
import pickle, random, string
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# Visualization
#from IPython.display import display


# In[2]:



# class PositionEmbedding(keras.layers.Layer):
#     def __init__(self,maxlen,embed_dim, *args, **kwargs):
#         super(PositionEmbedding, self).__init__(*args,**kwargs)
#         self.pos_emb = keras.layers.Embedding(input_dim=maxlen,
#                                               output_dim=embed_dim)
#         self.maxlen = maxlen
#         self.embed_dim = embed_dim
        
#     def call(self,x):
#         maxlen = tf.shape(x)[1]
#         print(maxlen)
#         positions = tf.range(start=0,limit=maxlen,delta=1)
#         positions = self.pos_emb(positions)
#         print(tf.shape(positions))
#         print(tf.shape(x))
#         return x + positions
    
#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'maxlen': self.maxlen,
#             'embed_dim': self.embed_dim 
#         })
#         return config


# In[3]:


class PositionEmbedding(keras.layers.Layer):
    def __init__(self,maxlen,embed_dim, *args, **kwargs):
        super(PositionEmbedding, self).__init__(*args,**kwargs)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen,
                                              output_dim=embed_dim)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        
    def call(self,x):
        maxlen = tf.shape(x)[1]
        print(maxlen)
        positions = tf.range(start=0,limit=maxlen,delta=1)
        positions = self.pos_emb(positions)
        print(tf.shape(positions))
        print(tf.shape(x))
        return x + positions
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'embed_dim': self.embed_dim 
        })
        return config


# In[4]:


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, *args, **kwargs):
        super(TransformerBlock, self).__init__(*args,**kwargs)
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                   key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="gelu"),
             keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate 

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads, 'embed_dim': self.embed_dim,
            'ff_dim': self.ff_dim, 'rate': self.rate
        })
        return config


# In[5]:


class MaskedTokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, *args, **kwargs):
        super(MaskedTokenAndPositionEmbedding, self).__init__(*args, **kwargs)
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size,
                                                output_dim=embed_dim,
                                                mask_zero=True)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen+1,
                                              output_dim=embed_dim,
                                              mask_zero=True)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=1, limit=maxlen+1, delta=1)
        positions = positions * tf.cast(tf.sign(x),tf.int32)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim
        })
        return config


# In[6]:


class MaskedTransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, *args, **kwargs):
        super(MaskedTransformerBlock, self).__init__(*args, **kwargs)
        self.att1 = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                    key_dim=embed_dim)
        self.att2 = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                    key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="gelu"),
             keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate 
        
    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.
        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, inputs, training):
        input_shape = tf.shape(inputs[0])
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        mask = self.causal_attention_mask(batch_size,
                                         seq_len, seq_len,
                                         tf.bool)
        # mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        attn_output1 = self.att1(inputs[0], inputs[0],
                                 attention_mask = mask)
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(inputs[0] + attn_output1)
        attn_output2 = self.att2(out1, inputs[1])
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layernorm1(out1 + attn_output2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm2(out2 + ffn_output)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads, 'embed_dim': self.embed_dim,
            'ff_dim': self.ff_dim, 'rate': self.rate
        })
        return config


# # Import outer decoder

# In[7]:


with open('../SavedModels/OuterTransformer/encoder_len53.json', 'r') as encoder_file, open('../SavedModels/OuterTransformer/decoder_len53.json', 'r') as decoder_file:
    encoder_json = encoder_file.read()
    decoder_json = decoder_file.read()
outer_encoder = keras.models.model_from_json(encoder_json,custom_objects={"PositionEmbedding":PositionEmbedding,"TransformerBlock":TransformerBlock,"MaskedTokenAndPositionEmbedding": MaskedTokenAndPositionEmbedding, "MaskedTransformerBlock":MaskedTransformerBlock})
outer_decoder = keras.models.model_from_json(decoder_json, custom_objects={"PositionEmbedding":PositionEmbedding,"TransformerBlock":TransformerBlock,"MaskedTokenAndPositionEmbedding": MaskedTokenAndPositionEmbedding, "MaskedTransformerBlock":MaskedTransformerBlock})
outer_encoder.load_weights("../SavedModels/OuterTransformer/encoder_len53.h5")
outer_decoder.load_weights("../SavedModels/OuterTransformer/decoder_len53.h5")
keras.utils.plot_model(outer_encoder, show_shapes=True)
keras.utils.plot_model(outer_decoder, show_shapes=True)


# # load input data

# In[8]:


# corpus = open('data/len5_10000-train.txt')#corpus = np.loadtxt(sys.argv[1], dtype=object)
# corpus = np.loadtxt(corpus, dtype=object)

# trainingSet = open('data/SG-10-train.txt')
# testingSet  = open('data/SG-10-test.txt')

# trainingSet = np.loadtxt(trainingSet, dtype=str)
# testingSet  = np.loadtxt(testingSet, dtype=str)


# In[9]:


corpus = open('data/len5_10000-train.txt')#corpus = np.loadtxt(sys.argv[1], dtype=object)
corpus = np.loadtxt(corpus, dtype=object)

# trainingSet = open('data/SG-10-train.txt')
# testingSet  = open('data/SG-10-test.txt')

trainingSet = np.loadtxt(sys.argv[1], dtype=str)
testingSet  = np.loadtxt(sys.argv[2], dtype=str)


# In[10]:


length = 10
padded_length = 20

embed_dim = 300 

# Note we are making these the same, but they don't -have- to be!
# input_length = padded_length
# output_length = padded_length

input_length = 5+1
output_length = 5+1

# Vocabulary sizes...
encoder_vocab_size = 30 # blank, a, b, c, ... j (bc/SG-10)
decoder_vocab_size = 30 # blank, a, b, c, ... j, start, stop


# In[11]:


# mapping function from characters to integers
def letter_to_int(char_array):
    # --- Create a dictionary for all the letters & start/stops ---
    alphabet = np.array([i for i in range(1, 31)]) # All letters plus STARTSENTENCE, STOPSENTENCE, start, stop
    mapping = dict()
    for i in range(len(alphabet) - 4):
        mapping[chr(ord('a') + i)] = alphabet[i]

    mapping['start'] = alphabet[26]
    mapping['stop']  = alphabet[27]
    mapping['STARTSETNENCE'] = alphabet[28]
    mapping['STOPSENTENCE']  = alphabet[29]
    
    # --- Map the characters in the input array to integers ---
    x_input = char_array
    x_input = [list(i) for i in x_input]
    X = []
    for word in x_input:
        X.append([mapping[sym] for sym in word])
        #print("word=", word)
    X = np.array(X)
    
    
    # --- Create Y, preY, postY ---
    Y = []
    for word in X:
        Y.append(np.concatenate((np.array([27]), word, np.array([28])), axis=0))

    Y = np.array(Y)
    preY  = Y[:, :-1]
    postY = Y[:, 1:]
    
    return X, Y, preY, postY, mapping


# In[12]:


# reverse mapping function 
def int_to_letter(encoding, mapping):
    enc_shape = encoding.shape
    
    flat_encoding = encoding.flatten() # Flatten array to just one dimension
    
    # list out keys and values separately
    key_list = list(mapping.keys())
    val_list = list(mapping.values())

    integers = []
    for letter in flat_encoding:
        integers.append(key_list[val_list.index(letter)])
        
    integers = np.array(integers)
    integers = np.reshape(integers, enc_shape)
    return integers


# In[13]:


X, Y, preY, postY, mapping = letter_to_int(corpus)


# In[14]:


# print("X.shape=", X.shape)
# print("Y.shape=", Y.shape)
# print("preY.shape=", preY.shape)
# print("postY.shape=", postY.shape)


# # Inner LSTM

# In[15]:


encoder_model = outer_encoder
decoder_model = outer_decoder


# In[16]:


#sample 10 indices from the coprus WOR
corIdx = np.random.randint(0,high=len(corpus),size=20)
#I could probably just do a np array of size 10000 
corIdx = np.random.choice(corIdx,size=10,replace=False)

outer_x = X[corIdx]
outer_preY = preY[corIdx]
outer_postY = postY[corIdx]

outer_embeddings = outer_encoder.predict(outer_x)

trainingSet_int = letter_to_int(trainingSet)[0]

testingSet_int = letter_to_int(testingSet)[0]

inner_x = np.array([outer_embeddings[trainingSet_int[x]-1] for x in range(len(trainingSet))]) #200 comes from the trainingSet 
inner_x_testing = np.array([outer_embeddings[testingSet_int[x]-1] for x in range(len(testingSet))])

#need testing inner_postY_testing


dog = np.zeros((1,5,300))
dog.shape
np.concatenate((dog, inner_x[0]))

#make inner_postY and outer_postY
dog = np.zeros((1,5,300))
inner_preY = np.array([np.concatenate((dog,inner_x[i])) for i in range(len(inner_x))])
inner_postY = np.array([np.concatenate((inner_x[i],dog)) for i in range(len(inner_x))])

#testing postY and preY
dog = np.zeros((1,5,300))
dog.shape
np.concatenate((dog, inner_x_testing[0]))

dog = np.zeros((1,5,300))
inner_preY_testing = np.array([np.concatenate((dog,inner_x_testing[i])) for i in range(len(inner_x_testing))])
inner_postY_testing = np.array([np.concatenate((inner_x_testing[i],dog)) for i in range(len(inner_x_testing))])


# In[17]:


#inner encoder construction 
hidden_size = 300
inner_encoder_input = keras.layers.Input(shape=(None,) + inner_x.shape[2:], name="inner_encoder_input")
#reshape layer here
inner_encoder_reshape = keras.layers.Reshape((-1,1500))(inner_encoder_input)
inner_encoder_hidden = keras.layers.LSTM(hidden_size, return_state=True, name="inner_encoder") #pass layers in here


# In[18]:


encoder_output, enc_state_h, enc_state_c = inner_encoder_hidden(inner_encoder_reshape)
encoder_states = [enc_state_h, enc_state_c]


# In[19]:


#inner decoder construction
inner_decoder_input_1 = keras.layers.Input(shape=(None,) + inner_preY.shape[2:], name="inner_dec_token_1")
inner_decoder_reshape_layer = keras.layers.Reshape((-1,1500))
inner_decoder_reshape = inner_decoder_reshape_layer(inner_decoder_input_1)
inner_decoder_input_2 = keras.layers.Input(shape=(None, 2), name="dec_start/stop")
inner_decoder_concat_layer = keras.layers.Concatenate()
inner_decoder_concat = inner_decoder_concat_layer([inner_decoder_reshape,inner_decoder_input_2])

inner_decoder_hidden = keras.layers.LSTM(hidden_size,return_sequences=True,return_state=True,name="inner_decoder") #need initial state for h and c


# In[20]:


#tie it together 
decoder_hidden_output, decoder_state_h, decoder_state_c = inner_decoder_hidden(inner_decoder_concat,
                                                                         initial_state=encoder_states) #below swap this with new input layer


# In[21]:


# decoder_dense_t1 = keras.layers.Dense(inner_encoder_reshape.shape[-1], activation='linear', name="token_1")(decoder_hidden_output)
# inner_output_reshape = keras.layers.Reshape((-1,inner_postY.shape[2],inner_postY.shape[3]))(decoder_dense_t1)
# decoder_dense_t2 = keras.layers.Dense(2, activation='sigmoid', name="start/stop")(decoder_hidden_output)


# In[22]:


inner_output_reshape_layer = keras.layers.Reshape((-1,inner_postY.shape[2], int(embed_dim / inner_postY.shape[2])))

inner_output_reshape = inner_output_reshape_layer(decoder_hidden_output)

decoder_dense_t1_layer = keras.layers.Dense(embed_dim, activation='linear', name="token_1")

decoder_dense_t1 = decoder_dense_t1_layer(inner_output_reshape)

decoder_dense_t2_layer = keras.layers.Dense(2, activation='sigmoid', name="start/stop")
decoder_dense_t2 = decoder_dense_t2_layer(decoder_hidden_output)


# In[23]:


inner_decoder_output = [decoder_dense_t1, decoder_dense_t2]


# In[24]:


inner_model = keras.Model([inner_encoder_input,inner_decoder_input_1, inner_decoder_input_2],
                         inner_decoder_output)


# In[25]:


#make the pre start/stop which will have dimensions (200,4,2)
#pre_start = np.array([[1,0,0,0] for x in range(0,200)])
#is it just the exact same as Blake's
s_s = {"start":[1,0],"stop":[0,1], "none":[0,0]}
pre_start = np.zeros((inner_x.shape[0], 4, 2))
post_stop = np.zeros((inner_x.shape[0], 4, 2))
pre_start[:,0,:] = s_s["start"]
post_stop[:,3,:] = s_s["stop"]


# In[26]:


inner_model.compile(loss = [keras.losses.MSE,keras.losses.binary_crossentropy],
               optimizer=keras.optimizers.Adam(),
               metrics=['accuracy'])


# In[27]:


#keras.utils.plot_model(inner_model,show_shapes=True)


# In[28]:


inner_model_input = {"input_1":inner_x,"input_2":inner_preY,
                     "dec_start/stop":pre_start}


# In[29]:


inner_model_target = {"inner_dec_token_1":inner_postY,"start/stop": post_stop}


# In[30]:


#try to train it 
inner_batch_size = 30
inner_epochs = 300
inner_history = inner_model.fit([inner_x,inner_preY, pre_start], [inner_postY,post_stop],
                         batch_size=inner_batch_size,
                         epochs=inner_epochs,
                         verbose=0,
                               validation_data=([inner_x_testing, inner_preY_testing, pre_start[0:100]],[inner_postY_testing, post_stop[0:100]]))


# In[31]:


#inner_history.history.keys()


# In[32]:


# plt.figure(1)  
# # summarize history for accuracy 
# plt.subplot(211)  
# plt.plot(inner_history.history['token_1_accuracy'])  
# plt.plot(inner_history.history['val_token_1_accuracy']) 
# plt.title('model accuracy')  
# plt.ylabel('reshape_2_accuracy')  
# plt.xlabel('epoch')  
# # summarize history for loss  
# plt.subplot(212)  
# plt.plot(inner_history.history['token_1_loss'])
# plt.plot(inner_history.history['val_token_1_loss'])  
# plt.title('model loss')  
# plt.ylabel('loss')  
# plt.xlabel('epoch')  
# plt.tight_layout()
# plt.show()  


# In[33]:


#accuracy = inner_model.evaluate([inner_x,inner_preY, pre_start], [inner_postY,post_stop])


# In[34]:


#accuracy = inner_model.evaluate([inner_x_testing,inner_preY_testing, pre_start[0:100]], [inner_postY_testing,post_stop[0:100]])


# # decouple the model

# In[35]:


#encoder 
inner_encoder_model = keras.Model(inner_encoder_input, encoder_states)


# In[36]:


#Decoder 
inner_decoder_state_input_h = keras.layers.Input(shape=(hidden_size,),
                                                name='inner_states_input_h')

inner_decoder_state_input_c = keras.layers.Input(shape=(hidden_size,),
                                                name='inner_states_input_c')


# In[37]:


# Connect hidden to input(s)
inner_decoder_states_input = [inner_decoder_state_input_h,
                             inner_decoder_state_input_c]

inner_decoder_hidden_output, inner_decoder_state_h, inner_decoder_state_c = inner_decoder_hidden(inner_decoder_concat,
              initial_state=inner_decoder_states_input)


inner_decoder_states = [inner_decoder_state_h, inner_decoder_state_c]

inner_output_reshape_2 = inner_output_reshape_layer(inner_decoder_hidden_output)


decoder_dense_t1 = decoder_dense_t1_layer(inner_output_reshape_2)#keras.layers.Dense(inner_encoder_reshape.shape[-1], activation='linear', name="token_1")
decoder_dense_t2 = decoder_dense_t2_layer(inner_decoder_hidden_output)#keras.layers.Dense(2, activation='sigmoid', name="start/stop")

#inner_output_reshape_2 = keras.layers.Reshape((-1,inner_postY.shape[2],inner_postY.shape[3]))(decoder_dense_t1(inner_decoder_hidden_output))

# Connect output to hidden(s)
inner_decoder_output = [decoder_dense_t1, decoder_dense_t2] #maybe a problem is here


# In[ ]:





# In[38]:


inner_decoder_model = keras.Model([inner_decoder_input_1, inner_decoder_input_2] + inner_decoder_states_input,
                                 inner_decoder_output + inner_decoder_states)


# In[39]:


#keras.utils.plot_model(inner_decoder_model, show_shapes= True)


# # without teacher forcing 

# In[40]:


#get the context
embedding = inner_encoder_model.predict(inner_x[0:1]) 
#without teacher forcing

input_tokens = np.zeros_like(inner_preY[0:1])

input_start = np.zeros_like(pre_start[0:1])

input_start[0,0:1,:] = pre_start[0,0:1,:]


for i in range(0,3): 
    output_tokens = inner_decoder_model.predict([input_tokens, input_start] + embedding)
    
    input_tokens[:,i+1:i+2,:,:] = output_tokens[0][:,i:i+1,:,:] #replace on input_tokens timestep
    input_start[:,i+1:i+2,:] = output_tokens[1][:,i:i+1,:]
    embedding = output_tokens[2:4]
output_tokens = inner_decoder_model.predict([input_tokens, input_start] + embedding)


# # Plug inner decoder's output into outer decoder

# In[41]:


#teacher forced result 
#output_tokens = inner_model.predict([inner_x[0:1], inner_preY[0:1], pre_start[0:1]])


# In[ ]:





# In[42]:


#output_tokens = [inner_postY[0:1], post_stop[0:1]]


arr = []

for k in range(0,3):
    context = output_tokens[0][0][k:k+1,:,:]
    i=0
    result = np.zeros_like(preY[i:i+1])
    result[0:1,0:1] = preY[i:i+1,0:1] # Start only...

    for j in range(output_length): #ouput length is 5+1 
        tokens = np.argmax(outer_decoder.predict([result,context]),-1)
        result[0:1,j+1:j+2] = tokens[0:1, j:j+1]
            
    result = tokens # Remove start token
    arr.append(int_to_letter(result, mapping)[0])
output = np.asarray(arr)


# In[43]:


output


# In[44]:


check = int_to_letter(outer_postY[trainingSet_int[0]-1], mapping)
check


# In[45]:


#int_to_letter(outer_x[trainingSet_int[0]-1], mapping)


# In[46]:


#word accuracy 
def word_accuracy(output,check):
    count = 0
    for i, j in zip(output, check):
            if(np.array_equal(i,j)):
                count +=1


    word_accuracy = count / len(output)
    return word_accuracy


# In[47]:


#letter accuracy 
def letter_accuracy(output, check):
    count = 0
    for i, j in zip(output, check):
            for x, y in zip(i,j):
                if(x == y):
                    count += 1


    letter_accuracy = count / (len(output[0]) * len(output))
    return letter_accuracy


# In[48]:


# word_accuracy(output, check)


# # Testing

# In[49]:


inner_x_testing.shape


# In[50]:


pre_start.shape


# In[51]:


word_acc_arr = []
letter_acc_arr = []
for m in range(0,10): #doing only 5 to save time but should be len(testingSet)
    #get the context
    embedding = inner_encoder_model.predict(inner_x_testing[m:m+1]) 
    #without teacher forcing

    input_tokens = np.zeros_like(inner_preY_testing[m:m+1])

    input_start = np.zeros_like(pre_start[m:m+1])

    input_start[0,0:1,:] = pre_start[m,0:1,:]

    for i in range(0,3):
        
        
#         output_tokens = inner_decoder_model.predict([input_tokens, input_start] + embedding)

#         input_tokens[:,i+1:i+2,:,:] = output_tokens[0][:,i:i+1,:,:] #replace on input_tokens timestep
#         input_start[:,i+1:i+2,:] = output_tokens[1][:,i:i+1,:]
#         embedding = output_tokens[2:4]
        
        output_tokens = inner_model.predict([inner_x_testing[m:m+1], input_tokens, input_start])
        input_tokens[:,i+1:i+2,:,:] = output_tokens[0][:,i:i+1,:,:] 
        input_start[:,i+1:i+2,:] = output_tokens[1][:,i:i+1,:]
                                       
    output_tokens = inner_model.predict([inner_x_testing[m:m+1], input_tokens, input_start])

    #output_tokens = [inner_postY_testing[m:m+1], post_stop[m:m+1]]
    #output_tokens = inner_decoder_model.predict([input_tokens, input_start] + embedding)

    
    
    
    
    #teacher forced result 
    #output_tokens = inner_model.predict([inner_x_testing[m:m+1], inner_preY_testing[m:m+1], pre_start[m:m+1]])
    
    arr = []

    for k in range(0,3):
        context = output_tokens[0][0][k:k+1,:,:]
        #context = inner_postY_testing[m][k:k+1,:,:]
        i=0
        input_tokens = np.zeros_like(preY[i:i+1])
        input_tokens[0:1,0:1] = preY[i:i+1,0:1] # Start only... and m

        for j in range(5): #ouput length is 5+1 
            result = np.argmax(outer_decoder.predict([input_tokens,context]),-1)
            input_tokens[0:1,j+1:j+2] = result[0:1, j:j+1] #0:1 m
        result = np.argmax(outer_decoder.predict([input_tokens,context]),-1) # Remove start token
        arr.append(int_to_letter(result, mapping)[0])

    output = np.asarray(arr)
    #print(output)

    check = int_to_letter(outer_postY[testingSet_int[m]-1], mapping)
    #print(check)

    word_acc_arr.append(word_accuracy(output, check))
    letter_acc_arr.append(letter_accuracy(output, check))


# In[52]:


output


# In[53]:


check


# In[54]:


final_word_acc = sum(word_acc_arr) / len(word_acc_arr)

final_letter_acc = sum(letter_acc_arr) / len(letter_acc_arr)


# In[55]:


print('''   Generalization Accuracy
-----------------------------
word_accuracy: %f
letter_accuracy: %f
'''%((final_word_acc*100), (final_letter_acc*100)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




