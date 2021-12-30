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

1st layer:The input layer, which has 11 nodes as depicted in Figure 1, is the first layer. The input is subsequently transmitted to the second layer.

2nd layer&3rd layer: As illustrated in Figure 1, the second layer of our ANN model is a hidden layer with six nodes. And the activation of each node in this layer is determined using an activation function, in this case a Rectified linear Unit (RELU). These generated activations are used as input for the 3rd hidden layer, which calculates each node's activation using the same activation functions.  

4th layer:  Because it offers us the result, the final layer is termed as the output layer. Because our problem is binary classification, we utilize a sigmoid activation function exclusively on the output layer. By employing sigmoid, we can readily interpret the output as probabilities because it has a constrained output between 0 and 1.

Step 4:  Hyperparameter-like batch size and epochs are initialized. Training data is fitted to the ANN classifier then, the exit status of the customer is predicted.  


Confusion Matrix

![C_m](https://user-images.githubusercontent.com/66326769/147728734-928c8a43-8e18-4459-9ffa-568524d28c7a.png)




Accuracy after 100 epochs

![Screenshot 2021-12-20 120013 - Copy](https://user-images.githubusercontent.com/66326769/147728240-32118a43-aa24-4c9a-aa4d-e80fbf7397d2.png)


![Screenshot 2021-12-20 120112](https://user-images.githubusercontent.com/66326769/147728262-ceadb802-9d71-47bc-b005-f902847a40f7.png)



Loss has decreased significantly as the number of epochs increased

![Figure 2021-12-30 123041](https://user-images.githubusercontent.com/66326769/147728214-78bc79dd-cb88-4139-9f2c-2a53beea4ac1.png)

