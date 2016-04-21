'''
LSTM realbook by Keunwoo Choi. (Keras 1.0)
Details on 
- repo:  https://github.com/keunwoochoi/lstm_real_book
- paper: https://arxiv.org/abs/1604.05358#
'''
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys


def sample(a, temperature=1.0):
	# helper function to sample an index from a probability array
	a = np.log(a) / temperature
	a = np.exp(a) / np.sum(np.exp(a))
	return np.argmax(np.random.multinomial(1, a, 1))

def get_model(maxlen, num_chars):
	# build an LSTM model, compile it, and return it
	print('Build model...')
	model = Sequential()
	model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, num_chars)))
	model.add(Dropout(0.2))
	model.add(LSTM(512, return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(num_chars))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model

def vectorize():
	print('Vectorization...')
	X = np.zeros((len(sentences), maxlen, num_chars), dtype=np.bool)
	y = np.zeros((len(sentences), num_chars), dtype=np.bool)
	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			X[i, t, char_indices[char]] = 1
		y[i, char_indices[next_chars[i]]] = 1
	return X, y

def main(character_mode):

	path = 'chord_sentences.txt' # the txt data source
	text = open(path).read()
	print('corpus length:', len(text))

	if character_mode:
		chars = set(text)
	else:
		chord_seq = text.split(' ')
		chars = set(chord_seq)
		text = chord_seq

	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))
	num_chars = len(char_indices)
	print('total chars:', num_chars)

	# cut the text in semi-redundant sequences of maxlen characters
	maxlen = 20
	step = 3
	sentences = []
	next_chars = []
	for i in range(0, len(text) - maxlen, step):
		sentences.append(text[i: i + maxlen])
		next_chars.append(text[i + maxlen])
	print('nb sequences:', len(sentences))

	# text to vectors
	X, y = vectorize()
	# build the model: stacked LSTM
	model = get_model(maxlen, num_chars)
	# train the model, output generated text after each iteration
	for iteration in range(1, 60):
		print()
		print('-' * 50)
		print('Iteration', iteration)
		with open(('result_iter_%02d.txt' % iteration), 'w') as f_write:

			model.fit(X, y, batch_size=512, nb_epoch=1)
			start_index = random.randint(0, len(text) - maxlen - 1)

			for diversity in [0.8, 1.0, 1.2]:
				print()
				print('----- diversity:', diversity)
				f_write.write('diversity:%4.2f\n' % diversity)
				if character_mode:
					generated = ''
				else:
					generated = []
				sentence = text[start_index: start_index + maxlen]
				seed_sentence = text[start_index: start_index + maxlen]
				
				if character_mode:
					generated += sentence
				else:
					generated = generated + sentence
				
				print('----- Generating with seed:')
				if character_mode:
					print(sentence)
					sys.stdout.write(generated)
				else:
					print(' '.join(sentence))

				if character_mode:
					num_char_pred = 1500
				else:
					num_char_pred = 150

				for i in xrange(num_char_pred):
					x = np.zeros((1, maxlen, num_chars))
					
					for t, char in enumerate(sentence):
						x[0, t, char_indices[char]] = 1.

					preds = model.predict(x, verbose=0)[0]
					next_index = sample(preds, diversity)
					next_char = indices_char[next_index]
					
					if character_mode:
						generated += next_char
						sentence = sentence[1:] + next_char
					else:
						generated.append(next_char)
						sentence = sentence[1:]
						sentence.append(next_char)

					if character_mode:
						sys.stdout.write(next_char)

					sys.stdout.flush()
				print()

				if character_mode:
					f_write.write(seed_sentence + '\n')
					f_write.write(generated)
				else:
					f_write.write(' '.join(seed_sentence) + '\n')
					f_write.write(' ' .join(generated))
				f_write.write('\n\n')
	return

if __name__=='__main__':
	main(character_mode=False)
