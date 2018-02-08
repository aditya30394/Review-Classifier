import math
import Vocabulary.vocabulary  # @UnresolvedImport
import Stopwords.stopwords  # @UnresolvedImport
import nltk
from nltk.probability import FreqDist


def multinomial_naive_bayes_unigram(training_file, test_file, stop_words):
    
    Vocabulary.vocabulary.build_vocabulary("../Stopwords/data_without_stopwords.txt")
    vocab_file = open("../Vocabulary/vocabulary.txt", "r")
    
    vocab_count = 0
    all_vocab = set()
    for line in vocab_file :
        words = line.split()
        all_vocab.add(words[0])
        vocab_count += 1
    
    separate_superdoc(training_file, stop_words)
    
    get_frequency("../Training/positive.txt",all_vocab)
    positive_superdoc = open("../Training/vocab_freq.txt", "r")
    positive_doc_count = 0
    
    with open("../Training/positive.txt") as positive_file:
        for line in positive_file:
            positive_doc_count = positive_doc_count + 1
    
    positive_vocab = {"+":positive_doc_count}
    
    for each_line in positive_superdoc:
        words_in_line = each_line.split()
        positive_vocab[words_in_line[0]] = words_in_line[2]
        
    positive_superdoc.close()
    
    get_frequency("../Training/negative.txt",all_vocab)
    negative_superdoc = open("../Training/vocab_freq.txt", "r")
    negative_doc_count = 0
    
    with open("../Training/negative.txt") as negative_file:
        for line in negative_file:
            negative_doc_count = negative_doc_count + 1
    
    negative_vocab = {"-":negative_doc_count}
    
    for each_line in negative_superdoc:
        words_in_line = each_line.split()
        negative_vocab[words_in_line[0]] = words_in_line[2]
        
    negative_superdoc.close()
    
    Stopwords.stopwords.remove_stopwords(test_file, stop_words,"../Training/testfile_without_stopwords.txt")
    test_handler = open("../Training/testfile_without_stopwords.txt","r")
    tp=0
    tn=0
    fp=0
    fn=0
    review_number=1;
    
    match = 0
    not_match = 0
        
        
    for each_review in test_handler:
        FLAG = 0;
        words_in_review = each_review.split()
        if words_in_review[0] == "+":
            FLAG = 1
        else:
            FLAG = 0
                
        prob_pos_given_review = predict_positive(words_in_review, positive_vocab, negative_vocab, vocab_count,all_vocab)
        prob_neg_given_review = predict_negative(words_in_review, positive_vocab, negative_vocab, vocab_count,all_vocab)
        REVIEW = 0
        
        if prob_pos_given_review < prob_neg_given_review:
            REVIEW = 0
        else:
            REVIEW = 1
        
       
        if FLAG==1 and REVIEW==1:
            tp=tp+1
            match = match + 1
        elif FLAG==1 and REVIEW==0:
            fn=fn+1
            not_match = not_match + 1
        elif FLAG==0 and REVIEW==1:
            fp=fp+1
            not_match = not_match + 1
        else:
            tn=tn+1
            match = match + 1
        
        print(str(review_number) + ". original = " + str(FLAG) +" prediction = " + str(REVIEW)+"\n")
        review_number = review_number + 1
        print("Match : " + str(match))
        print("Not Match : "+ str(not_match)+ "\n")
        
    accuracy = (tp+tn)/(tp+tn+fp+fn)                                    
    return accuracy

def predict_positive(words_in_review, positive_vocab, negative_vocab, vocab_count,all_vocab):
    prob_pos = math.log2(float(positive_vocab["+"])/(float(positive_vocab["+"]) + float(negative_vocab["-"])))  
    
    probability = prob_pos
    total_pos_vocab=0
    
    for key in positive_vocab:
        total_pos_vocab = total_pos_vocab + float(positive_vocab[key])
    
    total_pos_vocab = total_pos_vocab - float(positive_vocab["+"])
        
    for i in range(len(words_in_review)):
        if i==0:
            continue
        count=0
        if words_in_review[i] in positive_vocab:
            count = float(positive_vocab[words_in_review[i]])
            
        prob_word_given_pos = math.log2((count + 1)/(total_pos_vocab + vocab_count))
        if(words_in_review[i] in all_vocab):
            probability = probability + prob_word_given_pos
    return probability     


def predict_negative(words_in_review, positive_vocab, negative_vocab, vocab_count,all_vocab):
    
    prob_neg = math.log2(float(negative_vocab["-"])/(float(positive_vocab["+"]) + float(negative_vocab["-"])))
    
    probability = prob_neg
    total_neg_vocab=0
    
    for key in negative_vocab:
        total_neg_vocab = total_neg_vocab + float(negative_vocab[key])
    
    total_neg_vocab = total_neg_vocab - float(negative_vocab["-"])
        
    for i in range(len(words_in_review)):
        if i==0:
            continue
        count=0
        if words_in_review[i] in negative_vocab:
            count = float(negative_vocab[words_in_review[i]])
            
        prob_word_given_neg = math.log2((count + 1)/(total_neg_vocab + vocab_count))
        if(words_in_review[i] in all_vocab):
            probability = probability + prob_word_given_neg
    return probability     






def separate_superdoc(training_file, stop_words):
    
    Stopwords.stopwords.remove_stopwords(training_file, stop_words,"../Training/trainingfile_without_stopwords.txt")
    training_without_stopwords = open("../Training/trainingfile_without_stopwords.txt", "r")
    positive_superdoc = open("../Training/positive.txt", "w")
    negative_superdoc = open("../Training/negative.txt", "w")
    
    for line in training_without_stopwords:
        words_in_line = line.split();
        
        FLAG = 0
        
        for word in words_in_line :
            if word == '+' :
                positive_superdoc.write(word)
                FLAG = 1
                continue
            if word == '-' :
                negative_superdoc.write(word)
                FLAG = 0
                continue
            if FLAG == 1 :
                positive_superdoc.write(" " + word)
            else :
                negative_superdoc.write(" " + word)
            
        if FLAG == 1 :
            positive_superdoc.write("\n")
        else :
            negative_superdoc.write("\n")
            
    positive_superdoc.close()
    negative_superdoc.close()   
    return

def get_frequency(data_file,all_vocab):
    
    input_file = open(data_file, "r")
    input_file_contents = input_file.read()

    words = nltk.tokenize.word_tokenize(input_file_contents, 'english')
    fdist = FreqDist(words)
    print(fdist)

    output_file = open("../Training/vocab_freq.txt", "w")
    

    for word, frequency in fdist.most_common(4000):        
        if word in all_vocab  and word!='+' and word!='-':
            output_file.write(word + " : " + str(frequency) + "\n")        
            
    output_file.close()
    return 1

#data = "data.txt"
#stop_words = "stopwords.txt"

#accuracy= multinomial_naive_bayes_unigram(data, data, stop_words)
#print(accuracy)
#print("Separating Done!!")