import numpy as np
import os
import sys
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten, Activation, TimeDistributed
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import normalize

# Model type (char or word)
model_type = 'char'

# Print steps 
verbose = True

# Hyper Parameters
text_length = 100 # Length of the generated output in characters
seq_length = 50 # Number of characters read at a time
layers_count = 3 # Number of hidden layers
layers_dim = 700 # Hidden layer dimension   
dropout_layers = (1,) # Tuple of output layer locations
dropout_value = 0.3 # Dropout layer probabilities 
epoch_num = 1 # Number of epochs
batch_s = 100 # Batch size

# Filepaths
filepath = sys.path[0] + "\\_data\\ff - Wait, what.txt" # Training data
output_file = f"\\_outputs\\Output-{layers_count}-{layers_dim}-{epoch_num}.txt" # Output 

# Load specific model
load_model = False # Set this to true or false to change from load to gen
model_path = "\\_hdf5\\model-50-1.0042.hdf5" # Load model from hdf5 file

# Seed char vs seed text
# NOTE: Using seed text as an input makes text generation much slower
seed_text = True # If seed_text use some text as the first input to the RNN
seed_text_filepath = sys.path[0] + "\\_data\\HP P-stone 1st chapter excerpt.txt"
seed_char = 'h' # Otherwise use the seed character (we use h for harry)

def read_text(filepath):
    """
    Reads text file as utf8 and returns it as an all lowercase string.
    """

    text = open(filepath, encoding = "utf8").read()
    return text.lower()

def character_map(text):
    """
    Maps all characters to a unique integer.
    """

    print(f"Total character count: {len(text)}\n")

    characters = sorted(list(set(text))) # Get sorted list of individual characters
    n_to_char = {}
    char_to_n = {}

    num = 0
    for char in characters:
        n_to_char[num] = char
        char_to_n[char] = num
        num += 1

    return characters, n_to_char, char_to_n

def word_map(text):
    """
    Maps each word to a unique integer
    """

    # Replace puncation with words
    s = text.replace('.', " :period:")
    s = s.replace('\n', "")
    s = s.replace('"', " :quote:")
    s = s.replace(',', " :comma:")
    s = s.replace('?', " :quest:")

    words = sorted(set(s.split(" ")))

    n_to_word = {}
    word_to_n = {}

    num = 0
    for word in words:
        n_to_word[num] = word
        word_to_n[word] = num
        num += 1

    return words, n_to_word, word_to_n

def process_text(text, characters, n_to_char, char_to_n, length = seq_length):
    """
    Processes the input text into an array of sequences using the character maps
    """

    char_count = len(characters) # Number of individual characters
    dsl = len(text)//length # Number of possible sequences

    X = np.zeros((dsl, length, char_count))
    y = np.zeros((dsl, length, char_count))

    # For each sequence
    for i in range(0, dsl):
        X[i] = create_seqeunce_helper(i, text, dsl, char_count, char_to_n, 0, length)
        y[i] = create_seqeunce_helper(i, text, dsl, char_count, char_to_n, 1, length)

    return X, y, characters, n_to_char 

def create_seqeunce_helper(i, text, dsl, char_count, char_to_n, extra, length = seq_length):
    """
    Helper function for process text used to generate seqence arrays
    """

    seq_int = [] # Sequence mapped to integers
    output_seq = np.zeros((length, char_count)) # Output sequence which will become one item in input array  

    # Get the next sequence and map its characters to integers
    for v in text[i * length + extra : (i + 1) * length + extra]:
        # If the seed_text is missing a character we append 0
        if v in char_to_n:
            seq_int.append(char_to_n[v])
        else:
            seq_int.append(0)

    # For character in sequence
    for j in range(length):
        # Set column corrpsonding to that character to 1
        output_seq[j][seq_int[j]] = 1.0 

    return output_seq


def model(X, Y, characters):
    """
    Uses Keras to generate a RNN
    """

    model = Sequential()
    model.add(LSTM(layers_dim, input_shape=(None, len(characters)), return_sequences=True))
    model.add(Dropout(dropout_value))

    # Added layers_count hidden layers. More than 5 layers tends to result in over fitting   
    for i in range(layers_count - 1):
        model.add(LSTM(layers_dim, return_sequences = True))
        if(i + 1 in dropout_layers): # Add a dropout layer according to dropout_layers tuple
            model.add(Dropout(dropout_value))
    
    model.add(TimeDistributed(Dense(len(characters)))) # Add a dense layer to condense the output
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Load from saved model
    if load_model:
        model.load_weights(sys.path[0] + model_path)
        print("Model loaded.")
        return model

    # Only save model when loss function decreases
    filepath = sys.path[0] + "\\_hdf5\\model-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, Y, epochs=epoch_num, batch_size=batch_s, callbacks=callbacks_list)

    return model

def generate_text(model, length, text, X, characters, n_to_char, char_to_n, verbose = True, seed_text_str = None):
    """
    Generates output text. Given a seed character, uses that character to predict the next character.
    """

    # Start the model with a given character or text to act as the "seed"
    if verbose:
        print("Generating text...\n")
            
    input_arr = []
    output_arr = []

    if seed_text:
        # Read from seed text
        proc_tuple = process_text(seed_text_str, characters, n_to_char, char_to_n, length) 
        input_arr = proc_tuple[0]
        new_int_c = char_to_n[seed_text_str[0]]

    else:
        # Output will always start with the seed character
        new_int_c = char_to_n[seed_char] # Convert seed character to int
        output_arr = [n_to_char[new_int_c]]
        input_arr = np.ones((1, length, len(characters)))

    if verbose and len(output_arr) > 0:
        print(output_arr[-1], end="", flush=True)

    for i in range(length):
        input_arr[0, i, :][new_int_c] = 1 # Update X with last predicted char = 1
        new_int_c = np.argmax(model.predict(input_arr[:, : i + 1, :])[0], 1) # Predict the next char as int

        new_c = n_to_char[new_int_c[-1]] # Convert from int to char
        output_arr.append(new_c) # Append the predicted char to the sequence of predicted chars

        if verbose:
            print(new_c, end="", flush=True) # Print text as it is being generated

    # Build final result
    output = ""
    for c in output_arr:
        output += c

    return output

def train_and_generate(text_path):
    """
    Trains and generates the RNN according to the hyper-parameters at the top of the file.
    """

    print("\n------------------ ff.io Parameters ------------------")
    print(f"Generate text length: {text_length}")
    print(f"Sequence length: {seq_length}\n")
    print(f"{layers_count} layers with dimension {layers_dim}")
    print(f"{epoch_num} epochs with batch size {batch_s}\n")

    text = read_text(text_path)

    if load_model:
        print("Loading model from file.")

    if model_type == 'word':
        print("Creating word maps.")
        characters, n_to_char, char_to_n = word_map(text)
    
    else: # Default to character maps
        print("Creating character maps.")
        characters, n_to_char, char_to_n = character_map(text)

    if seed_text:
        seed_text_str = read_text(seed_text_filepath)

    print("Processing text.")
    X, Y, characters, n_to_char = process_text(text, characters, n_to_char, char_to_n)

    print("Modelling\n")
    mod = model(X, Y, characters)

    gen_text = generate_text(mod, text_length, text, X, characters, n_to_char, char_to_n, seed_text_str = seed_text_str)

    return gen_text

def main():
    gen_text = train_and_generate(filepath)

    print()
    print(gen_text)

    with open(output_file, "w") as text_file:
        text_file.write(gen_text)


if __name__ == "__main__":
    main()
