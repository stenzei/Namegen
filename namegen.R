# ------------------------------------------------------------------------------
# Based on the following example: 
# https://keras.rstudio.com/articles/examples/lstm_text_generation.html
# Note, different from the original example this code does not contain
# the for-loops for epochs and varying temperature. PArameters are defined
# as flags to allow hyperparameter tuning.
# ------------------------------------------------------------------------------

library(babynames)
library(dplyr)
library(keras)
library(readr)
library(purrr)
library(stringr)
library(tokenizers)


# Parameters -------------------------------------------------------------------

# Hperparameters
FLAGS <- flags(
  flag_integer("maxlen", 40, "Length of, for learning considered, character 
                              sequences.  I.e. max length of generated names. 
                              Affects data preparation."),
  flag_integer("inv_redundance", 3, "Inverted redundance: The data will be split 
                              into sequences, inv_redundance defines every which 
                              index of the original text a new sequence is 
                              started. I.e. inv_redundance <- 1 means that there 
                              are as much sequences as the length of the text.
                              inv_sequence <- maxlen means no redundancy. The 
                              text is split in sequences which do not overlap. 
                              Affects data preparation."),
  flag_integer("layer_lstm_units", 128, "Number of output units for the first 
                              layer, which is a lstm layer. It maps the input to 
                              this number of units, which become the input for 
                              the next layer. Affects the model."),
  flag_numeric("learning_rate", 0.01, "Learning rate for the optimizer. It 
                              defines the step size for going down the gradient. 
                              Affects the optimizer."),
  flag_integer("batch_size", 128, "The batch size, i.e. the number of samples 
                              per gradient update while training the model. 
                              Affects training."),
  flag_integer("number_of_epochs", 10, "The number of epochs to run the model.
                              Affects training."),
  flag_numeric("temperature", 0.3, "Weighting the predictions. The range of this 
                              parameter is [0,1]. The lower it is the higher is 
                              the probability that the predicted next character 
                              is the one with the highest probability according 
                              to the model. A temperature of 1 means that every 
                              character has the same probability. Affects 
                              prediction."),
  flag_integer("number_of_chars", 1000, "The number of characters to be 
                              generated after each epoch. Affects prediction.")
)


# Data Preparation -------------------------------------------------------------

# FIRST, create a vector containing all the baby names, 
# but split up into single characters. This vector we 
# call the 'text'.
babynames %>%                   # take babynames from babynames-package
  select(name) %>%              # select only the namrs column
  distinct() ->                 # consider every name only once
  baby_names                    # assign data to variable
  
baby_names[[1]] %>%             # take baby_names as vector (not as tibble) 
  str_to_lower() %>%            # all words are only in lower characters
  str_c(collapse = "\n") %>%    # collapse the vector with '/n' as seperator
  tokenize_characters(strip_non_alphanum = FALSE, # split up into chars
                      simplify = TRUE) -> # create a vector instead of a list
  text                          # assign data to variable

print(sprintf("corpus length: %d", length(text)))

# SECOND, create a vector of all characters
# contained in the text.
text %>%                        # take the text 
  unique() %>%                  # get all characters
  sort() ->                     # sort them
  unique_chars                  # assign data to a variable

print(sprintf("total chars: %d", length(unique_chars)))  

# THIRD, cut the text into semi-redundant 
# sequences of maxlen characters and store
# the characters following each sequence.
from <- 1                       # start with 1
to <- length(text) - FLAGS$maxlen - 1 # end where the last sequence starts
by <- FLAGS$inv_redundance      # start a sequence every inv_redundance'th index
sequence_start_indices <- seq(from, to, by)

sequence_start_indices %>%      # take the start indices
  map(~list(seqence = text[.x:(.x+FLAGS$maxlen-1)], # create a maxlen sequence
            next_char = text[.x+FLAGS$maxlen])) %>% # get char after sequence
  transpose() ->                # transpose the data
  dataset                       # assign data to a variable

# FOURTH, vectorize the data: X is a three-dimensional array,
# which contains a one-hot encoding for every character in every sequence.
# X is the input. y is a two-dimensional array, which contains for every 
# sequence one-hot encoded the character, which follows the sequence. y 
# represents the result.
X <- array(0, dim = c(length(dataset$seqence), FLAGS$maxlen, 
                      length(unique_chars)))
y <- array(0, dim = c(length(dataset$seqence), 
                      length(unique_chars)))

for (i in 1:length(dataset$seqence)) {
  unique_chars %>%              # take the unique_characters vector
    sapply(                     # apply to every element of that vector
      function(x) {             # the following: if the current character is
        as.integer(x == dataset$seqence[[i]]) # the same as the current one in
      }                         # the sequence, write '1'.
    ) -> X[i,,]                 # assign the resulting matrix to 'layer' i of X
  # do conceptionally the same for the result vector y:
  y[i,] <- as.integer(unique_chars == dataset$next_char[[i]])
}


# Model Definition -------------------------------------------------------------

# FIRST, clear the session to avoid unwanted 
# effects from playing around before.
k_clear_session()

# SECOND, define the model. We use a sequential model.
# I.e. the neural net is a linear stack of layers.
model <- keras_model_sequential()

# THIRD, define the structure of the model. 
# In this case, the net consists of three layers.
model %>%
  layer_lstm(FLAGS$layer_lstm_units,    # number of output units of first layer
             input_shape = c(FLAGS$maxlen, # dimensionality of the input of this
                             length(unique_chars))) %>% # layer
  layer_dense(length(unique_chars)) %>% # sec. layer with num. of (output) units
  layer_activation("softmax")           # apply softmax function to the output

# FORTH, define the optimizer. In this case it is
# rmsprop which works well for "mini" batches. It 
# takes the size of the gradient into account.
optimizer <- optimizer_rmsprop(lr = FLAGS$learning_rate)

# FIFTH, compile the model, i.e. configuring the 
# model for the following training.
model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer,
  metrics = c('accuracy')
)


# Training ---------------------------------------------------------------------

# Train the model.
model %>% fit(
  X, y,                           # pass input and output for learning
  batch_size = FLAGS$batch_size,  # define number of samples per gradient update
  epochs = FLAGS$number_of_epochs # set number of epochs
) -> history                      # keep track of the model history


# Prediction -------------------------------------------------------------------

# Helper function to compute an index of the unique_chars
# vector to find the next character, which then is appended
# to the result string. The higher temperature is, farther 
# afield from max(preds) is the return value of this function.
sample_mod <- function(preds, temperature = 1) {
  preds <- log(preds)/temperature
  exp_preds <- exp(preds)
  preds <- exp_preds/sum(exp_preds)
  
  rmultinom(1, 1, preds) %>% 
    as.integer() %>%
    which.max()
}

# Find a random sequence of characters to
# initially apply the trained model. This
# initial sequence can be viewed as seed.
start_index <- sample(1:(length(text) - FLAGS$maxlen), size = 1) # random index
sequence <- text[start_index:(start_index + FLAGS$maxlen - 1)]   # init. sequen.
generated <- ""                                                  # init result

# Apply the trained model to generate new names.
# number_of_chars defines the length of the  
# generated string, which containes the new names.
for (i in 1:FLAGS$number_of_chars) {
  # FIRST, prepare the sequence to be a 
  # valid input for the trained model.
  unique_chars %>%                      # take unique_chars and
    sapply(                             # create a one-hot-encoding matrice 
      function(x){                      # for the initially randomly chosen 
        as.integer(x == sequence)       # sequence;
      }                                 # reshape this matrix by adding one 
    ) %>%                               # dimension of size 1, to create a
    array_reshape(c(1, dim(.))) -> x    # valid input for the trained model

  # SECOND, apply the trained model to the 
  # prepared sequence x. The result is a vector 
  # giving the probability for each unique
  # character to be the next after the input
  # sequence.
  preds <- predict(model, x)
  
  # THIRD, determine the next character of the
  # resulting string. Allow for some variation:
  # the closer to 1 'temperature' is, the higher
  # is the probability to choose a less likely
  # character. (see function sample_mod)
  next_index <- sample_mod(preds, FLAGS$temperature)
  next_char <- unique_chars[next_index]
  
  # Append determined character to result string.
  generated <- str_c(generated, next_char, collapse = "")
  # Remove the first character from the sequence and
  # append the determined character to the sequence
  sequence <- c(sequence[-1], next_char)
}

cat(generated)
