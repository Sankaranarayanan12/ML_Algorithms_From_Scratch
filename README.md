## Machine Learning Algorithms (Linear Regression, Logistic Regression, Shallow Neural Network) implemented from scratch using numpy

### Thought Process and Learnings:
This project mainly helped me in learning the underlying math and mechanisms behind the different ML algorithms. It introduced me to the core workflow followed for an ML project which is:
1) ##### Data Preprocessing:
Includes reading the data, looking for missing values (if so fill it with mean or median of its column), convert it to suitable data type(numpy) for performing calculations on it , convert each data as proper numerical value which includes doing one-hot encoding for categorical data, splitting each value into different columns for multi valued columns, normalizing the data for better operation, splitting the dataset into train,CV and test dataset.


2) ##### Training the Model:
Includes defining the parameters, logic for gradient descent and finally getting optimal values for the parameters after performing the gradient descent for certain number of times with certain value of alpha.

3) ##### Testing the Model:
Test the model against the CV and test dataset and analyze its performance.

4) ##### Refining the Model:
Based on the results got in step 3 use appropriate measures to improve its performance.

-------

1) Linear Regression:
   For implementing linear regression , I took a sample dataset from kaggle which is "insurance.csv" where depending on various factors like age,sex,bmi,children,smoker,region a person's insurance cost can be predicted.
   #### What is Linear Regression ?
   Linear regression is an ML algorithm which is used to predict continuous numbers when a series of data is fed to it based upon adjusting its weights (W and b) by learning from each training example to optimize its weights properly in order to make good predictions.

   So I started this by first reading the csv file using pandas library functions. After this converted it to numpy in order to do matrix operations on the dataset. Performed binary encoding for sex,smoker and one-hot encoding for region columns.

   ##### Important Lesson Learnt:
   Before performing any mathematical calculations on the dataset it is necessary to convert it into np.float64 datatype.

   After this using a random number generator function and permutation function offered by numpy , randomized the order of the dataset. Split the dataset into train,CV and test and performed normalization using Z-score normalization.

   Now comes the gradient descent part, initialized the parameters W and b to zeros with appropriate dimensions and performing the gradient descent for certain number of iterations and learning rate adjusts the parameters to an optimized value.
   After this plotting the loss to see if it is following a declining trend which gives us a sign that the values are moving in the right direction.

   Further steps include using the optimized parameter values to predict values for CV and test data set and computing the Mean Squared errors. I have also used scikit learn's Linear Regression model trained on the similar setting and compared results against it.

 Here is the result I got:
 
 sklearn test MSE: 17167338.974685814 
 
  your test MSE:    17167577.333078675
  
--------------

2) Logistic regression: For implementing Logistic regression (Softmax) I used the datset from kaggle which is "date_fruit.csv" where depending on factors like Area , Perimeter,Major_Axis,Minor_axis etc the type of the datefruit can be predicted.
 #### What is Logistic Regression(Softmax)  ?
  Logistic Regression (Softmax) is an ML algorithm which is used for classification purposes. It is used to predict the class to which the given set of data belongs to (total 2 classes or more) by following a very similar pattern to that of Linear Regression. It also adjusts its parameters W and b by learning from many training examples in order to make it more optimized for predicting the correct class to which it belongs to.


I followed the similar process as was done for Linear Regression including reading csv file using pandas,converting it to numpy type,randomizing the order, converting to np.float64 type,splitting the data into train,CV and test, normalizing the data, doing one hot encoding for the target values.

After this I wrote the gradient descent logic , where first the parameters W and B were initialized to zeros with appropriate dimensions, performing the gradient descent for certain number of times and defined learning rate and finally at the end getting the optimized values for the parameters and plotting the loss curve as done before.

##### Important Lesson Learnt:
In order to improve the performance of the model it is best to try out a range of values for number of iterations and learning rate values in the CV dataset and pick the best one among them.

Finally followed the similar process of prediction against CV , test dataset calculating their accuracy and comparing against scikit learn's model.

Here is the result I got:

sklearn — train: 0.9602, cv: 0.8718, test: 0.9264

yours   — train: 0.9522, cv: 0.8974, test: 0.9264

------------

3) Shallow Neural Networks: For implementing Shallow Neural Networks , I used the sample dataset from kaggle which is "healthcare-dataset-stroke-data.csv" where depending upon factors like gender,age,hypertension etc , the likelihood of a stroke can be predicted.
   
#### What is Shallow Neural Network ?
   It is an ML model which consists of certain  hidden layers(1 in this case) and an output layer and each layer consists a fixed number of neurons where for each neuron in the hidden layer can be performing operations on the dataset using any of the activation functions including Linear Regression, ReLU, Logistic Regression etc and similarly for the output layer. So Neural Networks can be used for both regression and classification problems the only difference is it works well in more complex data and it is able to identify complex relationships among them and accordingly adjust its parameters by learning from training examples to an optimized one.

   Here also I followed the same initial steps including reading csv file using pandas, filling empty values,converting to numpy data type,performing one-hot encoding ,converting to np.float64 datatype,randomizing the order, splitting it into train,CV and test dataset and normalizing it .

   Then I wrote the gradient descent logic, where
   
  ##### Important Lesson Learnt:
   first the parameter W was initialized to random values instead of zeroes because each neuron should have its unique weight if all had zeros first then they would all be learning the same way resulting into a symmetry problem.

   and the parameter b was initialized to zeros with appropriate dimensions this is done for both the hidden layer and the output layer. Then gradient descent was performed for the output layer first and then back propagated into the hidden layer and this process continues for certain number of iterations and learning rate which are again chosen from experimenting with ranges of values and choosing the best among them by using F1 scores as a metric as this dataset has class imbalance.

   Finally at the end we get optimized value for W and b for both the hidden layer and the output layer and as followed for previous models the loss curve was plotted.

   Further steps included testing the model against CV and test dataset and comparing it against the scikit learn's model.

   Here is the result I got:
   
   From-Scratch vs sklearn (Test Set)
   
Precision - From-Scratch: 0.1711

Precision - sklearn:      0.129

Recall    - From-Scratch: 0.3611

Recall    - sklearn:      0.2222

F1        - From-Scratch: 0.2321

F1        - sklearn:      0.1633

   
   
   
