import numpy as np
import os
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten, Activation, TimeDistributed
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import normalize

# Model type (char or word)
model_type = 'char'

# Hyper Parameters
text_length = 100 # Length of the generated output in characters
seq_length = 50 # Number of characters read at a time
layers_count = 3 # Number of hidden layers
layers_dim = 700 # Hidden layer dimension   
epoch_num = 1 # Number of epochs
batch_s = 100 # Batch size

# Filepaths
filepath = sys.path[0] + "\\_data\\ff - top5_stories.txt" # Training data
output_file = f"Output-{layers_count}-{layers_dim}-{epoch_num}.txt" # Output 

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

def process_text(text, characters, n_to_char, char_to_n):
    """
    Processes the input text into an array of sequences using the character maps
    """

    char_count = len(characters) # Number of individual characters
    dsl = len(text)//seq_length # Number of possible sequences

    X = np.zeros((dsl, seq_length, char_count))
    y = np.zeros((dsl, seq_length, char_count))

    # For each sequence
    for i in range(0, dsl):
        X[i] = create_seqeunce_helper(i, text, dsl, char_count, char_to_n, 0)
        y[i] = create_seqeunce_helper(i, text, dsl, char_count, char_to_n, 1)

    return X, y, characters, n_to_char 

def create_seqeunce_helper(i, text, dsl, char_count, char_to_n, extra):
    """
    Helper function for process text used to generate seqence arrays
    """

    seq_int = [] # Sequence mapped to integers
    output_seq = np.zeros((seq_length, char_count)) # Output sequence which will become one item in input array  

    # Get the next sequence and map its characters to integers
    for v in text[i * seq_length + extra : (i + 1) * seq_length + extra]:
        seq_int.append(char_to_n[v])

    # For character in sequence
    for j in range(seq_length):
        # Set column corrpsonding to that character to 1
        output_seq[j][seq_int[j]] = 1.0 

    return output_seq


def model(X, Y, characters):
    """
    Uses Keras to generate a RNN
    """

    model = Sequential()
    model.add(LSTM(layers_dim, input_shape=(None, len(characters)), return_sequences=True))
    model.add(Dropout(0.3))

    # Added layers_count hidden layers. More than 5 layers tends to result in over fitting
    
    for i in range(layers_count - 2):
        model.add(LSTM(layers_dim, return_sequences = True))
        if(i == 1 or i == 3): # Add a dropout layer after the first and 3rd layer
            model.add(Dropout(0.3))
    
    # model.add(TimeDistributed(Dense(len(characters)))) # Add a dense layer to condense the output
    model.add(LSTM(layers_dim))
    model.add(Dense(len(characters))) # Add a dense layer to condense the output
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Only save model when loss function decreases
    filepath = sys.path[0] + "\\_hdf5\\model-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, Y, epochs=epoch_num, batch_size=batch_s, callbacks=callbacks_list)

    return model

def generate_text(model, length, X, characters, n_to_char, char_to_n, seed_char, verbose = True):
    """
    Generates output text. Given a seed character, uses that character to predict the next character.
    """

    # Start the model with a given character to act as the "seed"
    if verbose:
        print("Generating text...\n")

    new_int_c = char_to_n[seed_char] # Convert seed character to int

    # Output will always start with the seed character
    output_arr = np.array([n_to_char[new_int_c]])
    input_arr = np.zeros((1, length, len(characters)))

    for i in range(length):
        input_arr[0, i, :][new_int_c] = 1 # Update X with last predicted char = 1
        new_int_c = np.argmax(model.predict(input_arr[0, : i + 1, :])[0], 1) # Predict the next char as int

        new_c = n_to_char[new_int_c] # Convert from int to char
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

    if model_type == 'word':
        print("Creating word maps.")
        characters, n_to_char, char_to_n = word_map(text)
    
    else: # Default to character maps
        print("Creating character maps.")
        characters, n_to_char, char_to_n = character_map(text)

    print("Processing text.")
    X, Y, characters, n_to_char = process_text(text, characters, n_to_char, char_to_n)

    print("Modelling\n")
    mod = model(X, Y, characters)

    print("Generated Text:\n")
    gen_text = generate_text(mod, text_length, X, characters, n_to_char, char_to_n, 'h')

    return gen_text

def main():
    gen_text = train_and_generate(filepath)

    print()
    print(gen_text)

    with open(output_file, "w") as text_file:
        text_file.write(gen_text)


if __name__ == "__main__":
    main()
