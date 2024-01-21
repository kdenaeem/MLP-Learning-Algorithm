# Creating weather forecast using an Artificial Neural Network

( a lot of steps in this documentation are skipped, this is a just a basic outline )
This is a simple MLP algorithm that uses a gradient descent algorithm with Momentum and adjustable learning rates. 
This was fully written in python with libraries like Numpy, matplot lib to show the results and pandas 

# Context
The context I used to apply my ANN was San Diego's weather in 1987 - 1990. This is because the basis of my project was to predict the weather using past data, and this dataset was rich 
I wanted to gain a good amount of general background knowledge before I start to remove outliers and pick out data to delete. Checking the weather in San Diego over the course of the 3 years I found that San Diego had a general range of 5 degrees - 34 degrees. 
So, I applied a formatting function on my dataset and looked for numbers below 0 and found a few outliers. One was -21 and the other was -13 which were clearly outliers as over the course of 3 years in San Diego, not once did the temperature fall below 0. 
This was all done in excel to clean the data before it went through my algorithm. 

I also indexed the data to the data columns and randomised the values. Randomizing the data is important in order to prevent the network from learning patterns that are specific to the order of the training sample.
The data was also standardised because it lets you compare and analyze the data in a meaningful way even if the data has different units.

# Data separation 

![image](https://github.com/kdenaeem/MLP-Learning-Algorithm/assets/10659597/ac3a78e5-6718-454d-9d61-45840eb36a13)

## Implementation

<img width="351" alt="image" src="https://github.com/kdenaeem/MLP-Learning-Algorithm/assets/10659597/ec6c463f-1480-46e1-bfd4-b84c0c95e826">


Starting with 2 hidden nodes and 5 inputs

### Equation for Forward Propagation
<img width="222" alt="image" src="https://github.com/kdenaeem/MLP-Learning-Algorithm/assets/10659597/07a8d205-4313-41a9-ba2f-cac6380d3f03">

### Equation for Backward Propagation
<img width="222" alt="image" src="https://github.com/kdenaeem/MLP-Learning-Algorithm/assets/10659597/e4cd4fea-659b-4721-96e6-995bcafdd585">

Training and network selection

<img width="365" alt="image" src="https://github.com/kdenaeem/MLP-Learning-Algorithm/assets/10659597/4a2dd014-bb6b-483c-ba4e-0add1bd8ad93">

I determined that 2000 is a suitable number of epochs because anything more would waste time training the model, 
after this number of epochs training the model does not necessarily make much of a difference. 

## Trials

In order to learn from my model, I will be evaluating the influence from each hyperparameter below. 
Base Model : 
With 2 hidden nodes :  the minimum validations error is 0.003329 with a learning rate of 0.1. This is a very good model, however, it may be too simplistic to capture intrinsic patterns effectively. 
With 5 hidden nodes : The minimum validation error is 0.003278 with a learning rate of 0.1. This shows an improvement from 2 hidden layers however it is more complex than 2 hidden nodes. 
With 7 hidden nodes : The minimum validation error is 0.003184 with a learning rate of 0.05. This is comparable to 5 hidden nodes model however, the model with a learning rate 0.1 has a slightly higher validation error which suggests overfitting
Momentum model : 
With 2 hidden nodes :  the minimum validations error is 0.00340 with a learning rate of 0.1. 
With 5 hidden nodes : The minimum validation error is 0.003153 with a learning rate of 0.1. 
With 7 hidden nodes : The minimum validation error is 0.006682 with a learning rate of 0.01. This is inferior to other models with fewer hidden layers and could potentially be due to overfitting.

## Choosing an activation function
<img width="623" alt="image" src="https://github.com/kdenaeem/MLP-Learning-Algorithm/assets/10659597/15bc920a-17e0-45df-83d6-479ad1210a36">

From the table, we can see that the ReLu activation function performs the best with an MSE of 0.00424847 and sigmoid having the highest MSE, 
showing that it is the worst performing out of the three. 

# Evaluation of final model

<img width="623" alt="image" src="https://github.com/kdenaeem/MLP-Learning-Algorithm/assets/10659597/6fa6b6c5-2f83-4fb0-8f0a-5547d5fce30a">
In order to check the performance of my model I decided to plot a graph of the predicted dataset from testing data and the and the actual values from the testing data. 
The final model was Gradient descent with momentum and as you can see, the correlation of the graph is very close to the perfect prediction line. 








