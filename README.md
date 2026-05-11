# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY

Neural Network Regression is a supervised learning technique used to predict continuous numerical values from input data. In this experiment, a neural network model is developed to learn the relationship between one numeric input and one numeric output.

A neural network consists of input, hidden, and output layers connected through neurons. Hidden layers use activation functions such as ReLU to learn complex patterns in the data, while the output layer produces a continuous numerical value for regression.

Before training, the dataset is normalized using MinMaxScaler and split into training and testing data. The model is trained using the RMSprop optimizer and Mean Squared Error (MSE) loss function to reduce prediction error. Finally, the trained model is evaluated and used to predict output values for new input data.


## Neural Network Model
<img width="615" height="491" alt="Screen Shot 2026-04-20 at 14 45 34" src="https://github.com/user-attachments/assets/ba9b5e65-204c-4348-8538-e9a22c20a79b" />

## DESIGN STEPS

### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

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

### STEP 8: 

Use the trained model to predict  for a new input value.

## PROGRAM

### Name: Vikamuhan Reddy

### Register Number: 212223240181

```python

class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1,8)
    self.fc2 = nn.Linear(8,10)
    self.fc3 = nn.Linear(10,1)
    self.relu = nn.ReLU() # Added ReLU activation
    self.history = {'loss':[]}

  def forward(self,x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x
    
# initailze the model
lig = NeuralNet()
criterion = nn.MSELoss()
optimzer = optim.RMSprop(lig.parameters(),lr=0.0001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad() # Corrected typo from optimzer to optimizer
      loss = criterion(ai_brain(X_train),y_train)
      loss.backward()

      optimizer.step()
      lig.history['loss'].append(loss.item())


      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

### Dataset Information
<img width="619" height="131" alt="Screen Shot 2026-04-20 at 14 42 17" src="https://github.com/user-attachments/assets/674162c9-f7d4-4e9e-a2cf-0199c7b8fef4" />


### OUTPUT

### Training Loss Vs Iteration Plot
<img width="470" height="349" alt="Screen Shot 2026-04-20 at 14 42 34" src="https://github.com/user-attachments/assets/f992027a-c800-45e7-98bf-924274a1b881" />


### New Sample Data Prediction
<img width="592" height="105" alt="Screen Shot 2026-04-20 at 14 46 27" src="https://github.com/user-attachments/assets/db07a715-e672-4641-9db5-47ff885af92d" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
