import nltk
from nltk.probability import FreqDist


def build_vocabulary(data_file):
	
	input_file = open(data_file, "r")
	input_file_contents = input_file.read()

	words = nltk.tokenize.word_tokenize(input_file_contents, 'english')
	fdist = FreqDist(words)
	print(fdist)
	# print(fdist.most_common(2000))

	output_file = open("../Vocabulary/vocabulary.txt", "w")

	for word, frequency in fdist.most_common(4000):		
		if frequency >= 2 and word!='+' and word!='-':
			output_file.write(word + "\n")
			
	output_file.close()
	return 1

#if build_vocabulary("../Stopwords/data_without_stopwords.txt") :
#	print("Vocabulary built successfully!")
