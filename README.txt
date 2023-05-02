I used the libraries NumPy, Pandas and MatPlotLib for my work. To use the data, I mounted
my Google Drive account into the Google Colab.

Q2.1) Initially, there are the calculations of the number of the data instances 
belonging to classes. You can see the histograms displayed.


Q2.2 - Q2.3) These two uses the same functions, the only difference is the value of 
alpha. alpha equalts to 0 for Q2.2 and 1 for Q2.3.

SeparateClasses(): Its only input is the training set. I separated the training set 
according to the class labels. Afterwards, added the values in the same columns 
to each other (except for the 'class_label' column.) And add an additive column 
named 'summation' which represents the total number of words in that class. All 
these procedure aims to ease the process of calculation of likelihoods. Also, I 
created a list for prior probabilities of each class. This function returns the
merged dataset (class_arr), list of prior probablities (prior_list), and the list 
of classes (yk), which has the same order of the classes in the merged dataset such
as [3, 1, 2, 4, 0].


MNB(): The function for Multinomial Naive Bayes model. Using validation dataset, 
the alpha value (0 for the first model, 1 for the second model), and the outputs of 
SeparateClasses(), such as merged dataset, the list of prior probabilities and 
the list of indexes it calculates the posterior probabilities for each class 
and compare them. It choses the highest probability, and each time it determines
only one row's best fit. It scans the validation dataset row-wise and match the 
most suitable labels. It returns a list of predicted labels (predicted_labels).

metrics(): It inputs the list of predicted labels, which is the output of the
function MNB(), along with the validation dataset. Initially, I created a 
5x5 all-zeros array for the confusion matrix. Each row and column represents a 
different class, from class 0 to class 4 in an ascending order. It checks the actual
and predicted labels and increases the value at the corresponding index. Also, it 
keeps track of the number of the true predictions for the calculation of accuracy 
value and detection of the number of wrong predictions. It returns the confusion 
matrix (confusion_matrix), accuracy of the model (accuracy), and number of the wrong
predictions (wrong_pred).

Finally, I called the function metrics() and obtained the confusion matrix of the 
model,the accuracy value and number of predictions. I applied it for both of the 
questions. The results are displayed (also, between the functions, I printed the 
merged version of the training dataset for a better understanding.)

