# Multinomial Naive Bayes Classifier in Python to classify reviews as positive or negative

The project was done in 4 phases:
1. Collection of product reviews from e-commerce websites and labeling them as positive or negative. This collection was used as Training set.
2. Creation of "vocabulary" from the training set after removal of stop words like "is", "was", "he/she" etc.
3. Training a multinomial naive bayes model with laplace/add one smoothing. Both the unigram and bigram features were used for this model.
4. Ten fold cross validation to get accuracy measurements.

## Team Members
* Aditya Kumar
* Anand MP
* Sachin T Sany

## Accuracy using Unigram:
```
Test file - Accuracy(%)
data1.txt   74.54545454545455
data2.txt   72.72727272727273
data3.txt   63.63636363636364
data4.txt   69.09090909090909
data5.txt   74.54545454545455
data6.txt   74.54545454545455
data7.txt   67.27272727272727
data8.txt   65.45454545454545
data9.txt   49.09090909090909
data10.txt  63.79310344827587
```
## Average accuracy using Unigram:
**67.47%**

## Accuracy using Bigram:
```
Test file - Accuracy(%)
data1.txt   96.36363636363636
data2.txt   90.90909090909091
data3.txt   92.72727272727272
data4.txt   98.18181818181818
data5.txt   96.36363636363636
data6.txt   92.72727272727272
data7.txt   92.72727272727272
data8.txt   96.36363636363636
data9.txt   98.18181818181818
data10.txt  91.37931034482759
```

## Average accuracy using Bigram:
**94.59%**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
