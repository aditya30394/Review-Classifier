
import re

# remove_short_forms takes a line as argument and returns a line after removing the common short forms.
def remove_short_forms(line):           
    
    line = re.sub("'m", "am", line)
    line = re.sub("'s", "is", line)
    line = re.sub("'d", "would", line)
    line = re.sub("'ll", "will", line)
    line = re.sub("'ve", "have", line)
    line = re.sub("'re", "are", line)
    line = re.sub("won't", "would not", line)
    line = re.sub("doesn't", "does not", line)        
    line = re.sub("n't", "not", line)
    line = re.sub("would't", "would not", line)
    line = re.sub("'t", "not", line)
    line = re.sub(' [^A-Za-z ]+', '', line)
    
    return line

# remove_stopwords takes two files as argument and creates a new file outfile after removing the stopwords_set. 
def remove_stopwords(data, stopwords,outfile): 

    stopwords_set = set(line.strip() for line in open(stopwords))
#     print(stopwords_set)
 
    reviews_file = open(data, "r")
    output = open(outfile, "w")

    for line in reviews_file:
        
        line = remove_short_forms(line) # removes the common short forms from the line for consistency

        words_in_line = line.split();
        START_OF_REVIEW = 1;

        for word in words_in_line:
            if word == '+' and START_OF_REVIEW == 1:
                output.write(word)
                START_OF_REVIEW = 0
                continue
        
            elif word == '-' and START_OF_REVIEW == 1:
                output.write(word)
                START_OF_REVIEW = 0
                continue
                
            if word.lower() not in stopwords_set:
                    output.write(" " + word.lower()) # converts the word to lower case first, and then writes to the file.
            
        output.write("\n")
        
    reviews_file.close()
    output.close()
    return
    

#remove_stopwords("../Data/data.txt", "stopwords.txt","data_without_stopwords.txt")  
#print("Program Successfully Terminated!")
            