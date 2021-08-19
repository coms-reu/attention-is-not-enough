#!/usr/bin/python 
'''
July 2021
@authors: Joshua Miller & Dr. Joshua Phillips

Outer encoder decoder is a transformer; inner encoder/decoder is a transformer
'''
import sys
import numpy as np
import pickle, random, string
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf

strategy = tf.distribute.OneDeviceStrategy('gpu:1')

# --- Make transformer blocks for outer transformer ---
class OuterPositionEmbedding(keras.layers.Layer):
    def __init__(self,maxlen,embed_dim, *args, **kwargs):
        super(OuterPositionEmbedding, self).__init__(*args,**kwargs)
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
    
class OuterTransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, *args, **kwargs):
        super(OuterTransformerBlock, self).__init__(*args,**kwargs)
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
    
class OuterMaskedTokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, *args, **kwargs):
        super(OuterMaskedTokenAndPositionEmbedding, self).__init__(*args, **kwargs)
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
    
class OuterMaskedTransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, *args, **kwargs):
        super(OuterMaskedTransformerBlock, self).__init__(*args, **kwargs)
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
    
# --- Load outer model ---
path_enc_json = '../SavedModels/OuterTransformer/encoder_len53.json'
path_dec_json = '../SavedModels/OuterTransformer/decoder_len53.json'
path_enc_h5   = '../SavedModels/OuterTransformer/encoder_len53.h5'
path_dec_h5   = '../SavedModels/OuterTransformer/decoder_len53.h5'

with open(path_enc_json, 'r') as encoder_file, open(path_dec_json, 'r') as decoder_file:
    encoder_json = encoder_file.read()
    decoder_json = decoder_file.read()
outer_encoder = keras.models.model_from_json(encoder_json,custom_objects={"PositionEmbedding":OuterPositionEmbedding,"TransformerBlock":OuterTransformerBlock,"MaskedTokenAndPositionEmbedding": OuterMaskedTokenAndPositionEmbedding, "MaskedTransformerBlock":OuterMaskedTransformerBlock})
outer_decoder = keras.models.model_from_json(decoder_json, custom_objects={"PositionEmbedding":OuterPositionEmbedding,"TransformerBlock":OuterTransformerBlock,"MaskedTokenAndPositionEmbedding": OuterMaskedTokenAndPositionEmbedding, "MaskedTransformerBlock":OuterMaskedTransformerBlock})
outer_encoder.load_weights(path_enc_h5)
outer_decoder.load_weights(path_dec_h5)

# --- Load trainging and testing data ---
corpus = np.loadtxt(sys.argv[1], dtype=object)
trainingSet = np.loadtxt(sys.argv[2], dtype=object)
testingSet  = np.loadtxt(sys.argv[3], dtype=object)

# --- Functions to process trainging and testing data ---
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
    X = np.array(X)
    
    # --- Create Y, preY, postY ---
    Y = []
    for word in X:
        Y.append(np.concatenate((np.array([27]), word, np.array([28])), axis=0))

    Y = np.array(Y)
    preY  = Y[:, :-1]
    postY = Y[:, 1:]
    
    return X, Y, preY, postY, mapping

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

# --- Create embeddings ---
X, Y, preY, postY, mapping = letter_to_int(corpus)

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

# --- Make start and stop tokens ---
s_s = {"start":[1,0],"stop":[0,1], "none":[0,0]}
pre_start = np.zeros((inner_x.shape[0], 4, 2))
post_stop = np.zeros((inner_x.shape[0], 4, 2))
pre_start[:,0,:] = s_s["start"]
post_stop[:,3,:] = s_s["stop"]

# --- Transformer blocks for inner model ---
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
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

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class MaskedTransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(MaskedTransformerBlock, self).__init__()
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
    
class MaskedPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(MaskedPositionEmbedding, self).__init__()
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen+1,
                                              output_dim=embed_dim,
                                              mask_zero=True)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        
    def compute_output_shape(self, input_shape):
        return input_shape + (embed_dim,)
    
    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=1, limit=maxlen+1, delta=1)
        positions = positions * tf.cast(tf.sign(tf.math.count_nonzero(x,axis=2)),tf.int32)
        positions = self.pos_emb(positions)
        return x + positions

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def MaskedSparseCategoricalCrossentropy(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def MaskedSparseCategoricalAccuracy(real, pred):
    accuracies = tf.equal(tf.cast(real,tf.int64), tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# --- Model parameters ---
length = 10
padded_length = 20

# Note we are making these the same, but they don't -have- to be!
input_length = padded_length
output_length = padded_length

# Vocabulary sizes...
encoder_vocab_size = 30 # a, b, c, ... z, start, stop, STARTSENTENCE, STOPSENTENCE
decoder_vocab_size = 30 # a, b, c, ... z, start, stop, STARTSENTENCE, STOPSENTENCE


# Size of the gestalt, context representations...
embed_dim = 128  # Embedding size for each token (enc/dec inputs already embedded)
num_heads = 4  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
stack = 1
wd = 0.01


HIDDEN_SIZE = 300
BATCH_SIZE  = 50
EPOCHS      = 250

# --- Construct inner encoder/decoder ---
with strategy.scope():
    # Encoder
    encoder_input = keras.layers.Input(shape=(None,) + inner_x.shape[2:], name="inner_enc_token")

    encoder_reshape = keras.layers.Reshape((-1,1500))(encoder_input)

    encoder_embedding = keras.layers.Dense(embed_dim)(encoder_reshape)

    encoder_mask_pos_embedding = MaskedPositionEmbedding(maxlen=input_length,
                                                         embed_dim=encoder_embedding.shape[-1])(encoder_embedding)

    encoder_state = TransformerBlock(embed_dim=encoder_embedding.shape[-1],
                                     num_heads=num_heads,
                                     ff_dim=ff_dim)(encoder_mask_pos_embedding)
    encoder_model = keras.Model(encoder_input,encoder_state,name="InnerEncoder")

    # Decoder
    decoder_input = keras.layers.Input(shape=(None,) + inner_preY.shape[2:], name="inner_dec_token")

    decoder_context_input = keras.layers.Input(shape=encoder_state.shape[1:], name='inner_enc_state')

    decoder_reshape = keras.layers.Reshape((-1,1500))(decoder_input)

    decoder_startstop = keras.layers.Input(shape=(None, 2), name="dec_start/stop")

    decoder_inputs = [decoder_context_input, decoder_input, decoder_startstop]

    decoder_concat = keras.layers.Concatenate()([decoder_reshape, decoder_startstop])

    decoder_embedding = keras.layers.Dense(embed_dim)(decoder_concat)

    decoder_mask_pos_embedding = MaskedPositionEmbedding(maxlen=inner_preY.shape[1],
                                                         embed_dim=decoder_embedding.shape[-1])(decoder_embedding)

    decoder_block = MaskedTransformerBlock(embed_dim=decoder_mask_pos_embedding.shape[-1],
                                           num_heads=num_heads,
                                           ff_dim=ff_dim)

    decoder_hidden_output = decoder_block([decoder_mask_pos_embedding, decoder_context_input])

    x = keras.layers.Dense(inner_postY.shape[2]*embed_dim)(decoder_hidden_output)

    inner_output_reshape = keras.layers.Reshape((-1,inner_postY.shape[-2],embed_dim))(x)

    decoder_dense_t1 = keras.layers.Dense(inner_postY.shape[-1], activation='linear', name="output_token")(inner_output_reshape)

    decoder_dense_startstop = keras.layers.Dense(2, activation='sigmoid', name="start/stop")(decoder_hidden_output)

    decoder_outputs = [decoder_dense_t1, decoder_dense_startstop]

    decoder_model = keras.Model(decoder_inputs,decoder_outputs,name="InnerDecoder")

    # Tie encoder and decoder into one model
    #with strategy.scope():
    #model = keras.Model([encoder_input]+ decoder_inputs, decoder_outputs)
    coupled_inputs = [keras.layers.Input(encoder_model.inputs[0].shape[1:]),
                      keras.layers.Input(decoder_model.inputs[1].shape[1:]),
                      keras.layers.Input(decoder_model.inputs[2].shape[1:])]                     
    coupled_outputs = decoder_model([encoder_model(coupled_inputs[0])] + coupled_inputs[1:])
    model = keras.Model(coupled_inputs, coupled_outputs)

    # --- Compile and fit model ---
    model.compile(loss = [keras.losses.MSE,keras.losses.binary_crossentropy],
               optimizer=keras.optimizers.Adam(),
               metrics=['accuracy'])

    model_input = {"inner_enc_token":inner_x, "inner_dec_token":inner_preY,
                         "dec_start/stop":pre_start}
    model_target = {"output_token":inner_postY, "start/stop": post_stop}

    #with strategy.scope():
    history = model.fit([inner_x,inner_preY, pre_start], [inner_postY,post_stop],
                         batch_size=BATCH_SIZE,
                         epochs=EPOCHS,
                         verbose=0)

# --- Make functions to decypher model's output ---
def word_accuracy(output,check):
    count = 0
    for i, j in zip(output, check):
            if(np.array_equal(i,j)):
                count +=1


    word_accuracy = count / len(output)
    return word_accuracy

def letter_accuracy(output, check):
    count = 0
    for i, j in zip(output, check):
            for x, y in zip(i,j):
                if(x == y):
                    count += 1


    letter_accuracy = count / (len(output[0]) * len(output))
    return letter_accuracy

# --- Get accuracy ---
with strategy.scope():
    word_acc_arr = []
    letter_acc_arr = []
    for m in range(100):
        #get the context
        context = encoder_model.predict(inner_x_testing[m:m+1]) 
        #without teacher forcing

        input_tokens = np.zeros_like(inner_preY_testing[m:m+1])

        input_start = np.zeros_like(pre_start[m:m+1])

        input_start[0,0:1,:] = pre_start[0,0:1,:]

        for i in range(0,3):

            output_tokens = decoder_model.predict([context, input_tokens, input_start])

            input_tokens[:,i+1:i+2,:,:] = output_tokens[0][:,i:i+1,:,:] #replace on input_tokens timestep
            input_start[:,i+1:i+2,:] = output_tokens[1][:,i:i+1,:]
            #context = output_tokens[2:4]
        output_tokens = decoder_model.predict([context, input_tokens, input_start])

        arr = []

        for k in range(3):
            context = output_tokens[0][0][k:k+1,:,:]
            actual_context = inner_postY_testing[m][k:k+1,:,:]
            # Use this if you just want to test the outer decoder...
            context = actual_context
            i=0
            tokens = np.zeros_like(outer_preY[0:1])
            tokens[0:1,0:1] = outer_preY[0:1,0:1] # Start only...

            for j in range(5): #ouput length is 5+1
                result = np.argmax(outer_decoder.predict([tokens,context]),-1)
                tokens[0:1,j+1:j+2] = result[0:1, j:j+1]
            result = np.argmax(outer_decoder.predict([tokens,context]),-1)
            arr.append(int_to_letter(result, mapping)[0])

        output = np.asarray(arr)

        check = int_to_letter(outer_postY[testingSet_int[m]-1], mapping)

        word_acc_arr.append(word_accuracy(output, check))
        letter_acc_arr.append(letter_accuracy(output, check))
# use `model.metrics_names` to get indices for accuracy:
print('-----------------------------')
print('''   Generalization Accuracy
-----------------------------
word_accuracy: %f
letter_accuracy: %f
'''%(sum(word_acc_arr) / len(word_acc_arr) * 100, sum(letter_acc_arr) / len(letter_acc_arr) * 100))