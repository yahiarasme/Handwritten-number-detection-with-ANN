import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle


"""
There are 60000 photos in the training, and 10000 photos for testing, each one with  28x28 pixel,
with values from 0 to 255.  And there are 60000 labels for the training photos, and 10000 label for testing,
with values  from 0 to 9 
"""  
# read traning data set 
train_image = pd.read_csv('D:\project python\data set\csvTrainImages 60k x 784.csv')
train_label = pd.read_csv('D:\project python\data set\csvTrainLabel 60k x 1.csv')

# read testing data set
test_image = pd.read_csv('D:\project python\data set\csvTestImages 10k x 784.csv')
test_label = pd.read_csv('D:\project python\data set\csvTestLabel 10k x 1.csv')

"""
The data is between 0 and 255, we will wrap these value to be in 
the range 0.01 to 1 by  multiplying each pixel by 0.99 / 255 and adding 0.01 to the result.
This way, we avoid 0 values  as inputs, which are capable of preventing weight updates.
We are ready now to turn our labelled images into one-hot representations.
Instead of  zeros and one, we create 0.01 and 0.99, which will be better for our calculations 
"""

# normalization
fac = 0.99 / 255
train_image = np.asfarray(train_image) * fac + 0.01
test_image = np.asfarray(test_image) * fac + 0.01
train_label = np.asfarray(train_label)
test_label = np.asfarray(test_label)

# one hot encoding
train_targets = np.array(train_label).astype(np.int)
train_labels_one_hot = np.eye(np.max(train_targets) + 1)[train_targets]
test_targets = np.array(test_label).astype(np.int)
test_labels_one_hot = np.eye(np.max(test_targets) + 1)[test_targets]

train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99
print(train_label[0])          # 1.0
print(train_labels_one_hot[0])  # [0.01 0.99 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01]

"""
Build activation function :
3.1 Sigmoid for the Hidden Layer 1 
 
3.2 softmax for the Hidden Layer 2  It squish each input 𝑥𝑖 between 0 and 1 and normalizes the values to give
a  proper probability distribution where the probabilities sum up to one 
"""

# build ANN 

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)


# visualization
def view_classify(img, ps):
    ps = np.squeeze(ps)
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols = 2)
    ax1.imshow(img.reshape(28, 28))
    ax1.set_title(ps.argmax())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

"""
3.1 Input Layer  For the 28x28 photos we create a 784 node for each pixel    
3.2 Hidden Layer 1  256 nodes    
3.3 Hidden Layer 2  128 nodes    
3.4 Output Layer  In Output there are 10 nodes, each for a number from 0 to 9  
"""
    
class NeuralNetwork:
    def __init__(self):
        
        self.lr = 0.001
        
        self.w1 = np.random.randn(784, 256) 
        self.b1 = np.zeros((1, 256)) 
        
        self.w2 = np.random.randn(256, 128)
        self.b2 = np.zeros((1, 128)) 
        
        self.w3 = np.random.randn(128, 10)
        self.b3 = np.zeros((1, 10))

"""
=> Forward Propagation is to take the inputs, multiply by the weights (random numbers) 
=> Back propagation is to upgrade the weight, using a ​loss function​ represent the fail in  our training  
"""
        
    def feedforward(self):        
        
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)

        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)

        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)
        
    def backprop(self):
        output_errors = self.y - self.a3

        a3_delta = output_errors * (self.a3 * (1.0 - self.a3))  # w3
        
        z2_delta = np.dot(output_errors, self.w3.T)
        a2_delta = z2_delta * (self.a2 * (1.0 - self.a2)) # w2
        
        z1_delta = np.dot(z2_delta, self.w2.T)
        a1_delta = z1_delta * (self.a1 * (1.0 - self.a1)) # w1
        
        self.w3 += self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 += self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        
        self.w2 += self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 += self.lr * np.sum(a2_delta, axis=0, keepdims=True)
        
        self.w1 += self.lr * np.dot(self.x.T, a1_delta)
        self.b1 += self.lr * np.sum(a1_delta, axis=0, keepdims=True)
        
"""
One pass through the entire database is called ​epoch​, for each there is a training pass  calculate the loss,
do backward pass and update weights and bias.
"""
    def train(self, x, y):
        
        '''input_vector and target_vector can 
        be tuple, list or ndarray'''
        
        self.x = np.array(x, ndmin=2)
        self.y = np.array(y, ndmin=2)
        
        self.feedforward()
        self.backprop()
    
    def predict(self, data):
        self.x = np.array(data, ndmin=2)
        self.feedforward()
        return self.a3
    
    def confusion_matrix(self, x, y): 
        cm = np.zeros((10, 10), int)
        for i in range(len(x)): 
            res = self.predict(x[i]) 
            res_max = res.argmax()
            target = y[i] 
            cm[res_max, int(target)] += 1 
        return cm 

    def precision(self, y, confusion_matrix):
        col = confusion_matrix[:, y] 
        return confusion_matrix[y, y]/col.sum() 

    def recall(self, y, confusion_matrix): 
        row = confusion_matrix[y, :] 
        return confusion_matrix[y, y]/row.sum() 

    def evaluate(self, x, y): 
        corrects, wrongs = 0, 0 
        for i in range(len(x)): 
            res = self.predict(x[i]) 
            res_max = res.argmax() 
            if res_max == y[i]: 
                corrects += 1 
            else: 
                wrongs += 1 
        return corrects, wrongs

model = NeuralNetwork()

epochs = 10

for epoch in range(epochs):  
    print("epoch: ", epoch+1)
    for i in range(len(train_image)):
        model.train(train_image[i], train_labels_one_hot[i])
        
    corrects, wrongs = model.evaluate(train_image, train_label)
    print("accruracy train: ", corrects / ( corrects + wrongs))
    corrects, wrongs = model.evaluate(test_image, test_label)
    print("accruracy test :", corrects / ( corrects + wrongs))
    print("="*50)
    cm = model.confusion_matrix(train_image, train_label)
    for i in range(10):
        print("digit: ", i, "precision :", model.precision(i, cm))
        print("digit: ", i, "recall :", model.recall(i, cm))
        print("="*80)
        
        
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb')) 






