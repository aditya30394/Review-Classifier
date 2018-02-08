import Bigram.bigm  # @UnresolvedImport

def generate_training_file(filenames,testfile):
    with open('trainingfile.txt','w') as trainingfile:
        for file in filenames:
            if(file!=testfile):
                with open(file) as infile:
                    for line in infile:
                        trainingfile.write(line)
    return 


filenames = ['data1.txt','data2.txt','data3.txt','data4.txt','data5.txt','data6.txt','data7.txt','data8.txt','data9.txt','data10.txt']
accuracies=[]

for testfile in filenames:
    generate_training_file(filenames, testfile)  
    accuracy= Bigram.bigm.multinomial_naive_bayes_unigram_bigram('../Stopwords/data_without_stopwords.txt',testfile,'../Stopwords/stopwords.txt')
    print("Accuracy: ")                
    print(accuracy)
    accuracies.append(accuracy)  
              
avg_accuracy= float(sum(accuracies)/len(accuracies))
print(avg_accuracy)