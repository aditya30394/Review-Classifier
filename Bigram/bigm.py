import string
from test.test_asyncio.test_events import data_file
from shutil import copyfile
import Vocabulary.vocabulary  # @UnresolvedImport
import Training.training  # @UnresolvedImport
import Stopwords.stopwords  # @UnresolvedImport
from builtins import int
import math


# from Training.training import predict_negative

def multinomial_naive_bayes_unigram_bigram(training_file, test_file, stop_words):
    Vocabulary.vocabulary.build_vocabulary("../Stopwords/data_without_stopwords.txt")
    # Creates data with start and stop added to it.
    add_start_stop('../Stopwords/data_without_stopwords.txt', "data_with_start_stop.txt")

    all_bigram_vocab = bigram_vocab("data_with_start_stop.txt", "bigramVocabulary.txt")
    vocab_bigram_file = open("bigramVocabulary.txt", "r")

    vocab_count = 0
    for line in vocab_bigram_file:
        #         words = line.split()
        #         all_vocab.add(words[0])
        vocab_count += 1

    vocab_file = open("../Vocabulary/vocabulary.txt", "r")
    # we get all unigrams here. Reusing the old ones

    unigram_vocab = set()
    for line in vocab_file:
        words = line.split()
        unigram_vocab.add(words[0])
    # create the positive and negative super docs
    Training.training.separate_superdoc(training_file, stop_words)

    Training.training.get_frequency("../Training/positive.txt", unigram_vocab)
    positive_superdoc = open("../Training/vocab_freq.txt", "r")
    positive_doc_count = 0

    with open("../Training/positive.txt") as positive_file:
        for line in positive_file:
            positive_doc_count = positive_doc_count + 1

    positive_vocab = {"+": positive_doc_count}

    for each_line in positive_superdoc:
        words_in_line = each_line.split()
        positive_vocab[words_in_line[0]] = words_in_line[2]

    positive_superdoc.close()

    positive_unigram_vocab = positive_vocab;

    add_start_stop('../Training/positive.txt', "data_with_start_stop_positive.txt")

    positive_bigram_vocab = bigram_vocab("data_with_start_stop_positive.txt", "positiveBigramVocabulary.txt")

    #     print(positive_bigram_vocab)
    #     print(positive_unigram_vocab)

    positive_unigram_vocab = getCombinedVocabCount(positive_bigram_vocab, positive_unigram_vocab)
    #     print(positive_unigram_vocab)

    # ------------------------------Negative Vocab build starts here-----------------------------------

    Training.training.get_frequency("../Training/negative.txt", unigram_vocab)
    negative_superdoc = open("../Training/vocab_freq.txt", "r")
    negative_doc_count = 0

    with open("../Training/negative.txt") as negative_file:
        for line in negative_file:
            negative_doc_count = negative_doc_count + 1

    negative_vocab = {"-": negative_doc_count}

    for each_line in negative_superdoc:
        words_in_line = each_line.split()
        negative_vocab[words_in_line[0]] = words_in_line[2]

    negative_superdoc.close()

    negative_unigram_vocab = negative_vocab;

    add_start_stop('../Training/negative.txt', "data_with_start_stop_negative.txt")

    negative_bigram_vocab = bigram_vocab("data_with_start_stop_negative.txt", "negativeBigramVocabulary.txt")

    #     print(positive_bigram_vocab)
    #     print(negative_unigram_vocab)

    negative_unigram_vocab = getCombinedVocabCount(negative_bigram_vocab, negative_unigram_vocab)
    #     print(negative_unigram_vocab)

    # ---------------------------------Prediction Starts Here-------------------------------------------------------
    Stopwords.stopwords.remove_stopwords(test_file, stop_words, "../Training/testfile_without_stopwords.txt")
    add_start_stop("../Training/testfile_without_stopwords.txt", "testfile_with_start_stop.txt")
    test_handler = open("../BigramEvaluation/testfile_with_start_stop.txt", "r")
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    review_number = 1;

    match = 0
    not_match = 0

    for each_review in test_handler:
        FLAG = 0;
        words_in_review = each_review.split()
        if words_in_review[0] == "+":
            FLAG = 1
        else:
            FLAG = 0

        prob_pos_given_review = predict_positive(words_in_review, positive_unigram_vocab, positive_bigram_vocab,
                                                 negative_unigram_vocab, negative_bigram_vocab, vocab_count)
        prob_neg_given_review = predict_negative(words_in_review, positive_unigram_vocab, positive_bigram_vocab,
                                                 negative_unigram_vocab, negative_bigram_vocab, vocab_count)
        REVIEW = 0

        if prob_pos_given_review < prob_neg_given_review:
            REVIEW = 0
        else:
            REVIEW = 1

        if FLAG == 1 and REVIEW == 1:
            tp = tp + 1
            match = match + 1
        elif FLAG == 1 and REVIEW == 0:
            fn = fn + 1
            not_match = not_match + 1
        elif FLAG == 0 and REVIEW == 1:
            fp = fp + 1
            not_match = not_match + 1
        else:
            tn = tn + 1
            match = match + 1

        print(str(review_number) + ". original = " + str(FLAG) + " prediction = " + str(REVIEW) + "\n")
        review_number = review_number + 1
        print("Match : " + str(match))
        print("Not Match : " + str(not_match) + "\n")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


# Positive probability prediction starts here

def predict_positive(words_in_review, positive_unigram_vocab, positive_bigram_vocab, negative_unigram_vocab,
                     negative_bigram_vocab, vocab_count):
    prob_pos = math.log2(
        float(positive_unigram_vocab["+"]) / (float(positive_unigram_vocab["+"]) + float(negative_unigram_vocab["-"])))

    probability = prob_pos
    total_pos_vocab = 0

    for key in positive_unigram_vocab:
        if float(positive_unigram_vocab[key]) > 0:
            total_pos_vocab = total_pos_vocab + float(positive_unigram_vocab[key])
        else:
            total_pos_vocab = total_pos_vocab + 0

    for key in positive_bigram_vocab:
        total_pos_vocab = total_pos_vocab + float(positive_bigram_vocab[key])

    total_pos_vocab = total_pos_vocab - float(positive_unigram_vocab["+"])

    DO_NOTHING = 0
    for i in range(len(words_in_review) - 1):
        if i == 0:
            LAST_NOT_PRESENT = 0
            continue
        count = 0

        word1 = words_in_review[i]
        word2 = words_in_review[i + 1]

        bigram = (word1, word2)

        if bigram in positive_bigram_vocab:
            if (positive_bigram_vocab[bigram] >= 2):
                count = float(positive_bigram_vocab[bigram])
                LAST_NOT_PRESENT = 0
            else:
                if LAST_NOT_PRESENT == 0:
                    if word1 == 'START':
                        if word2 in positive_unigram_vocab:
                            count = float(positive_unigram_vocab[word2])
                        else:
                            count = 0
                    else:
                        if word2 == 'STOP':
                            if word1 in positive_unigram_vocab:
                                count = float(positive_unigram_vocab[word1])
                            else:
                                count = 0
                        else:
                            if word2 in positive_unigram_vocab:
                                count = float(positive_unigram_vocab[word2])
                            else:
                                count = 0
                    LAST_NOT_PRESENT = 1

                else:
                    if word2 == 'STOP':
                        DO_NOTHING = 1
        if DO_NOTHING:
            prob_word_given_pos = 0
        else:
            if count < 0:
                count = 0
            prob_word_given_pos = math.log2((count + 1) / (total_pos_vocab + vocab_count))
        probability = probability + prob_word_given_pos
    return probability


def predict_negative(words_in_review, positive_unigram_vocab, positive_bigram_vocab, negative_unigram_vocab,
                     negative_bigram_vocab, vocab_count):
    prob_neg = math.log2(
        float(negative_unigram_vocab["-"]) / (float(positive_unigram_vocab["+"]) + float(negative_unigram_vocab["-"])))

    probability = prob_neg
    total_neg_vocab = 0

    for key in negative_unigram_vocab:
        if float(negative_unigram_vocab[key]) > 0:
            total_neg_vocab = total_neg_vocab + float(negative_unigram_vocab[key])
        else:
            total_neg_vocab = total_neg_vocab + 0

    for key in negative_bigram_vocab:
        total_neg_vocab = total_neg_vocab + float(negative_bigram_vocab[key])

    total_neg_vocab = total_neg_vocab - float(negative_unigram_vocab["-"])

    DO_NOTHING = 0
    for i in range(len(words_in_review) - 1):
        if i == 0:
            LAST_NOT_PRESENT = 0
            continue
        count = 0

        word1 = words_in_review[i]
        word2 = words_in_review[i + 1]

        bigram = (word1, word2)

        if bigram in negative_bigram_vocab:
            if (negative_bigram_vocab[bigram] >= 2):
                count = float(negative_bigram_vocab[bigram])
                LAST_NOT_PRESENT = 0
            else:
                if LAST_NOT_PRESENT == 0:
                    if word1 == 'START':
                        if word2 in negative_unigram_vocab:
                            count = float(negative_unigram_vocab[word2])
                        else:
                            count = 0
                    else:
                        if word2 == 'STOP':
                            if word1 in negative_unigram_vocab:
                                count = float(negative_unigram_vocab[word1])
                            else:
                                count = 0
                        else:
                            if word2 in negative_unigram_vocab:
                                count = float(negative_unigram_vocab[word2])
                            else:
                                count = 0
                    LAST_NOT_PRESENT = 1

                else:
                    if word2 == 'STOP':
                        DO_NOTHING = 1
        if DO_NOTHING:
            prob_word_given_neg = 0
        else:
            if count < 0:
                count = 0
            prob_word_given_neg = math.log2((count + 1) / (total_neg_vocab + vocab_count))
        probability = probability + prob_word_given_neg
    return probability


def getCombinedVocabCount(bigram_vocab_dict, unigram_vocab_dict):
    for bigram in bigram_vocab_dict:
        if (bigram_vocab_dict[bigram] >= 2):
            (word1, word2) = bigram
            # if word1 in unigram_vocab_dict:
            #     unigram_vocab_dict[word1] = int(unigram_vocab_dict[word1]) - int(bigram_vocab_dict[bigram])
            if word2 in unigram_vocab_dict:
                unigram_vocab_dict[word2] = int(unigram_vocab_dict[word2]) - int(bigram_vocab_dict[bigram])
                #             print ("just checking: " + unigram_vocab_dict[word1])
                #             print ("check 2" + str(bigram_vocab_dict[bigram]))
    return unigram_vocab_dict


def bigram_vocab(input_file, output_file):
    copyfile('../Vocabulary/vocabulary.txt', output_file);
    vocab_file = open(output_file, 'a');
    inputfile = open(input_file, "r")
    fileContents = inputfile.read()
    bigrams = {}
    words_punct = fileContents.split()
    words = words_punct
    bigram_count = 0
    for index, word in enumerate(words):
        if index < len(words) - 1:
            # we only look at indices up to the
            # next-to-last word, as this is
            # the last one at which a bigram starts
            w1 = words[index]
            w2 = words[index + 1]
            # bigram is a tuple,
            # like a list, but fixed.
            # Tuples can be keys in a dictionary
            bigram = (w1, w2)
            if bigram in bigrams:
                bigrams[bigram] = bigrams[bigram] + 1
            else:
                bigrams[bigram] = 1

    # sort bigrams by their counts
    bigrams[('+', 'START')] = 0
    bigrams[('-', 'START')] = 0
    bigrams[('STOP', '-')] = 0
    bigrams[('STOP', '+')] = 0
    sorted_bigrams = sorted(bigrams.items(), key=lambda pair: pair[1], reverse=False)

    #     Print the bigrams
    #     for bigram, count in sorted_bigrams:
    #         print (bigram, ":", count)


    for bigram in bigrams:
        (word1, word2) = bigram
        if bigrams[bigram] >= 2:
            vocab_file.write(word1 + ' ' + word2 + '\n');
            bigram_count += 1
        else:
            bigrams[bigram] = 0

    print(bigram_count)

    vocab_file.close();
    inputfile.close();
    return bigrams


def add_start_stop(input_file, output_file):
    data_file = open(input_file, "r")
    output = open(output_file, "w")

    for line in data_file:

        words_in_line = line.split();
        START_OF_REVIEW = 1;

        for word in words_in_line:
            if (word == '+' or word == '-') and START_OF_REVIEW == 1:
                output.write(word + " START")
                START_OF_REVIEW = 0
                continue

            output.write(" " + word.lower())  # converts the word to lower case first, and then writes to the file.

        output.write(" STOP\n")

    data_file.close()
    output.close()

    return 1

    # if multinomial_naive_bayes_unigram_bigram("../Data/data.txt", "../Data/data.txt", "../Stopwords/stopwords.txt"):
    #     print ("Done! ")
