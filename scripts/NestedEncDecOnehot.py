'''
July 2021
@author: Joshua Miller

Blake's nested LSTM encoder-decoder
'''
#!usr/bin/python3 

import numpy as np
import pickle, random, string
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf

strategy = tf.distribute.OneDeviceStrategy('gpu:1')

path_enc_json = 'SavedModels/outer_encdec_intembed/encoder_len5_J_10000.json'
path_dec_json = 'SavedModels/outer_encdec_intembed/decoder_len5_J_10000.json'
path_enc_h5   = 'SavedModels/outer_encdec_intembed/encoder_len5_J_10000.h5'
path_dec_h5   = 'SavedModels/outer_encdec_intembed/decoder_len5_J_10000.h5'

with open(path_enc_json, 'r') as encoder_file, open(path_dec_json, 'r') as decoder_file:
    encoder_json = encoder_file.read()
    decoder_json = decoder_file.read()
outer_encoder = keras.models.model_from_json(encoder_json)
outer_decoder = keras.models.model_from_json(decoder_json)
outer_encoder.load_weights(path_enc_h5)
outer_decoder.load_weights(path_dec_h5)

corpus = np.loadtxt(sys.argv[1], dtype=object) #open('../thesis-master/data/len5_10000-train.txt')#
trainingSet = np.loadtxt(sys.argv[2], dtype=object) #open('../thesis-master/data/SG-10-train.txt')
testingSet  = np.loadtxt(sys.argv[3], dtype=onject)#open('../thesis-master/data/SG-10-test.txt')

# --- Function to help with decoding the model's output later ---
def ProcessOutput(array):
    array = np.array(array)
    index = np.argmax(array, axis=-1).flatten()[0]
    array[0, 0, index] = 1
    new_array = np.floor(array)
    return new_array

# --- Create a mapping of words from the corpus to the roles ---
encoded_mapping = {}
selected_words = {}
for letter in string.ascii_lowercase[:10]:
    # Store the letter with the word for use in testing
    onehot = encode.onehot(random.choice(corpus))
    selected_words[letter] = np.concatenate((onehot.copy(), encode.onehot("stop").reshape(1,28)))
    encoded_mapping[letter] = outer_encoder.predict(np.array([onehot]))
    
# --- Prep roles using the encodings created above ---
roles   = trainingSet #argv[2]    
x_train = []
for sentence in roles:
    x_train.append([encoded_mapping[letter] for letter in sentence])
x_train = np.array(x_train) # shape (n, 3, 2, 1, 50)

LENGTH_IDK  = x_train.shape[-1] # Replacing '50' in the code

t1 = x_train[:,:,0,0,:] # new shape (n,3,50)
t2 = x_train[:,:,1,0,:] # " '' "
# 4 time steps. pre
pre_t1 = np.concatenate((np.zeros((x_train.shape[0],1,LENGTH_IDK)), t1), axis = 1) # Orig. (x_train.shape[0], 1, 50)
pre_t2 = np.concatenate((np.zeros((x_train.shape[0],1,LENGTH_IDK)), t2), axis = 1)
post_t1 = np.concatenate((t1, np.zeros((x_train.shape[0],1,LENGTH_IDK))), axis = 1)
post_t2 = np.concatenate((t2, np.zeros((x_train.shape[0],1,LENGTH_IDK))), axis = 1)

# Start or stop tokens
s_s = {"start": [0,1], "stop": [1,0], "none": [0,0]}
pre_t3 = np.zeros((x_train.shape[0], 4, 2))
post_t3 = np.copy(pre_t3)
pre_t3[:,0,:] = s_s["start"]
post_t3[:,3,:] = s_s["stop"]

# --- Hyperparameters ---
HIDDEN_SIZE = 300
BATCH_SIZE  = 100
EPOCHS      = 1600

# --- Construct inner encoder/decoder with teacher forcing ---
with strategy.scope():
    encoder_input_t1 = keras.layers.Input(shape=(None, t1.shape[2]), name="enc_token_1")
    encoder_input_t2 = keras.layers.Input(shape=(None, t1.shape[2]), name="enc_token_2")
    encoder_input = keras.layers.Concatenate()([encoder_input_t1, encoder_input_t2])

    encoder_hidden = keras.layers.LSTM(HIDDEN_SIZE, return_state=True, name="Hidden encoder")
    # Tie them together
    encoder_output, enc_state_h, enc_state_c = encoder_hidden(encoder_input)
    # Don't need the encoder outputs, just need the states.
    encoder_states = [enc_state_h, enc_state_c]
    
     decoder_input_t1 = keras.layers.Input(shape=(None, t1.shape[2]), name="dec_token_1")
    decoder_input_t2 = keras.layers.Input(shape=(None, t1.shape[2]), name="dec_token_2")
    decoder_input_t3 = keras.layers.Input(shape=(None, 2), name="dec_start/stop")
    decoder_input = keras.layers.Concatenate()([decoder_input_t1, decoder_input_t2, decoder_input_t3])

    decoder_hidden = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True, name="Hidden decoder")
    # Tie it together
    decoder_hidden_output, decoder_state_h, decoder_state_c = decoder_hidden(decoder_input,
                                                                             initial_state=encoder_states)
    decoder_dense_t1 = keras.layers.Dense(t1.shape[2], activation='linear', name="token_1")
    decoder_dense_t2 = keras.layers.Dense(t1.shape[2], activation='linear', name="token_2")
    decoder_dense_t3 = keras.layers.Dense(2, activation='sigmoid', name="start/stop")
    # Connect output to hidden
    decoder_output = [decoder_dense_t1(decoder_hidden_output), decoder_dense_t2(decoder_hidden_output), decoder_dense_t3(decoder_hidden_output)]
    
    model = keras.Model([encoder_input_t1, encoder_input_t2, decoder_input_t1, decoder_input_t2, decoder_input_t3], decoder_output)
    
     model.compile(loss = [keras.losses.MSE, keras.losses.MSE, keras.losses.binary_crossentropy],
                   optimizer=keras.optimizers.Adam(),
                   metrics=['accuracy'])

    model_input = {"enc_token_1": t1, "enc_token_2": t2, "dec_token_1": pre_t1, "dec_token_2": pre_t2, "dec_start/stop": pre_t3}
    model_target = {"token_1": post_t1, "token_2": post_t2, "start/stop": post_t3}
    
    # --- Train model ---
    history = model.fit(model_input, model_target,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=0)
    accuracy = model.evaluate(model_input, model_target) # use `model.metrics_names` to get indices for accuracy:
    #print('T1 Accuracy:', accuracy[4]*100.0, '%')
    #print('T2 Accuracy:', accuracy[5]*100.0, '%')
    #print('T3 Accuracy:', accuracy[6]*100.0, '%')
    
    # --- Restructure model without teacher forcing ---
    encoder_model = keras.Model([encoder_input_t1, encoder_input_t2], encoder_states)

    # Decoder
    decoder_state_input_h = keras.layers.Input(shape=(HIDDEN_SIZE,), name="states_input_h")
    decoder_state_input_c = keras.layers.Input(shape=(HIDDEN_SIZE,), name="states_input_c")
    # inputs to hidden
    decoder_states_input = [decoder_state_input_h, decoder_state_input_c]
    decoder_hidden_output, decoder_state_h, decoder_state_c = decoder_hidden(decoder_input,
                                                                             initial_state=decoder_states_input)
    decoder_states = [decoder_state_h, decoder_state_c]
    # hidden to outputs

    decoder_output = [decoder_dense_t1(decoder_hidden_output), decoder_dense_t2(decoder_hidden_output), decoder_dense_t3(decoder_hidden_output)]
    decoder_model = keras.Model(
        [decoder_input_t1, decoder_input_t2, decoder_input_t3] + decoder_states_input,
        decoder_output + decoder_states)
    
    # --- Get output from inner and outer decoders ---
    x_test = []
    correct_result = [] # used to get accuracy at end
    roles = testingSet
    for sentence in roles:
        x_test.append([encoded_mapping[letter] for letter in sentence])
        correct_result.append([selected_words[letter] for letter in sentence])
    x_test = np.array(x_test) # shape (n, 3, 2, 1, 50)
    correct_result = np.array(correct_result)
    t1 = x_test[:,:,0,0,:] # new shape (n,3,50)
    t2 = x_test[:,:,1,0,:] # " '' "
    # 4 time steps. pre
    pre_t1 = np.concatenate((np.zeros((x_test.shape[0],1,LENGTH_IDK)), t1), axis = 1)
    pre_t2 = np.concatenate((np.zeros((x_test.shape[0],1,LENGTH_IDK)), t2), axis = 1)

    # Start tokens
    pre_t3 = np.zeros((x_test.shape[0], 4, 2))
    pre_t3[:,0,:] = s_s["start"]

    outer_result = np.empty((len(x_test),3,6,28))
    for i, sentence in enumerate(x_test):
        context = encoder_model.predict({"enc_token_1": t1[i:i+1], "enc_token_2": t2[i:i+1]})
        dec_t1 = np.zeros((1,1,LENGTH_IDK))
        dec_t2 = np.zeros((1,1,LENGTH_IDK))
        dec_s_s = pre_t3[0:1,0:1,:]
        inner_result = np.zeros([4,2,LENGTH_IDK])
        output_length = 3

        # obtain the result from the inner decoder
        for x in range(output_length+1):
            out1, out2, out3, h, c = decoder_model.predict({"states_input_h": context[0], 
                                             "states_input_c": context[1],
                                             "dec_token_1": dec_t1,
                                             "dec_token_2": dec_t2,
                                             "dec_start/stop": dec_s_s})
            context = [h,c]
            dec_t1 = out1
            dec_t2 = out2
            dec_s_s = out3
            inner_result[x,0,:] = out1
            inner_result[x,1,:] = out2

        # obtain the result from the outer decoder
        output_length = 5
        for word in range(3):
            context = []
            context.append(inner_result[word,0:1,:])
            context.append(inner_result[word,1:2,:])
            token = np.array(encode.onehot("start"))
            token = token.reshape([1, 1, token.shape[0]])
            for letter in range(output_length + 1):
                out, h, c = outer_decoder.predict([token] + context)
                token = ProcessOutput(out)
                context = [h,c]
                outer_result[i, word, letter, :] = token
                
# --- Get word and letter level accuracy ---
word_accuracy = 0
letter_accuracy = 0
for answer, response in zip(correct_result, outer_result):
    # check each word
    for word in range(3):
        if np.array_equal(answer[word,:,:], response[word,:,:]):
            word_accuracy += 1
            letter_accuracy += 6
        #check each letter
        else:
            for letter in range(6):
                if np.array_equal(answer[word,letter,:], response[word,letter,:]):
                    letter_accuracy += 1
                    
word_accuracy /= float(correct_result.shape[0] * 3)
letter_accuracy /= float(correct_result.shape[0] * 3 * 6)

print('-----------------------------')
print('''   Generalization Accuracy
-----------------------------
word_accuracy: %f
letter_accuracy: %f
'''%((word_accuracy*100), (letter_accuracy*100)))