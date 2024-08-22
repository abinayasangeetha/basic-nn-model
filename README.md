# Developing a Neural Network Regression Model

## AIM:

To develop a neural network regression model for the given dataset.

## THEORY:
   Neural networks are composed of basic input and output components called neurons, inspired by biological neurons in the brain. These neurons are linked together, with each connection having an associated weight that influences the output.Regression is used to determine the relationship between a dependent variable and one or more independent variables. The effectiveness of regression models depends on how well the regression equation fits the given data. However, in most cases, these models do not perfectly align with the data.To begin, import the necessary libraries and load the dataset. Check the data types of the columns. Next, divide the dataset into training and testing sets. We will then create a neural network with three hidden layers, using the ReLU activation function for each layer. Once the network is set up, we will train it on the dataset and make predictions based on the trained model.

## Neural Network Model

![Screenshot 2024-08-22 234030](https://github.com/user-attachments/assets/64942e2f-5d35-40ca-af05-45d12fd9b9cb)

## DESIGN STEPS

### STEP 1:
Loading the dataset

### STEP 2:
Split the dataset into training and testing

### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:
Build the Neural Network Model and compile the model.

### STEP 5:
Train the model with the training data.

### STEP 6:
Plot the performance plot

### STEP 7:
Evaluate the model with the testing data.

## PROGRAM
``` Name:ABINAYA S
 Register Number: 212222230002
```

## Importing Required packages
```py
from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default
import pandas as pd
```

## Authenticate the Google sheet
```py
auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)
worksheet = gc.open('Mydata').sheet1
data = worksheet.get_all_values()
```
## Construct Data frame using Rows and columns
```py
dataset1=pd.DataFrame(data[1:], columns=data[0])
dataset1=dataset1.astype({'x':'float'})
dataset1=dataset1.astype({'y':'float'})
dataset1.head(20)
X=df[['x']].values
Y=df[['y']].values
```
## Split the testing and training data
```py
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1=Scaler.transform(X_train)
```
## Build the Deep learning Model
```py
ai_brain=Sequential([
    Dense(8,activation = 'relu',input_shape=[1]),
    Dense(10,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train.astype(np.float32),epochs=2000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```

## Evaluate the Model
```py
X_test1=Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1=[[19]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```



## Dataset Information


## OUTPUT

### Training Loss Vs Iteration Plot

Include your plot here

### Test Data Root Mean Squared Error

Find the test data root mean squared error

### New Sample Data Prediction

Include your sample input and output here

## RESULT

Include your result here
