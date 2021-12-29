# Artificial_Neural_Network

![ann](https://user-images.githubusercontent.com/66326769/147590402-08d772b0-25d6-4f43-b32a-ea2a87476e4d.png)

Fig1: Artificial Neural Network architecture

The architecture of the neural network is [11,6,6,1] with: 

•	11 independent variables. 
•	2 hidden layers with 6 nodes on each hidden layer
•	Since we have classification problem, we will have one node in output layer  

ANN alorithm

Step 1: The dataset is imported and preprocessed. Preprocessing is needed to get a good quality result. The dataset is split suitable for training and testing. We have 80% for training and 20% for testing. 

Step 2: Initializing the Ann 
A sequential model from Keras API is used. It is a linear stack of layers and a commonly used architecture.

Step 3: We then add various layers to the architecture 
1st layer: The first layer consists of the input layer which consists of 11 nodes as shown in   Fig1 The input is then passed to the layer 2nd  
2nd layer&3rd layer: The second layer in our ANN model is a hidden layer consisting of 6 nodes as shown in Fig1. And for each node in this layer activation is calculated using an activation function which in this case is Rectified linear Unit (RELU). These activations calculated are the input for the 3rd hidden layer that uses the same activation functions to calculate the activation of each node.   
4th layer:  Because it offers us the result, the final layer is termed the output layer. Because our problem is binary classification, we utilize a sigmoid activation function exclusively on the output layer. By employing sigmoid, we can readily interpret the output as probabilities because it has a constrained output between 0 and 1.

Step 4:  Hyperparameter-like batch size and epochs are initialized. Training data is fitted to the ANN classifier then, the exit status of the customer is predicted.  
