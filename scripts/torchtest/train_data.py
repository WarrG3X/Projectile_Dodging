import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing



TEST_SIZE = 80
TRAIN_SIZE = 400 - TEST_SIZE

np.random.seed(0)
torch.manual_seed(0)

#Load Data from CSV File
data = np.loadtxt('data.csv',delimiter=',',dtype=np.float32)
#scaler = preprocessing.StandardScaler().fit(X_train)
np.random.shuffle(data)
X = Variable(torch.from_numpy(data[:TRAIN_SIZE,0:-1]))
X_test = Variable(torch.from_numpy(data[TRAIN_SIZE:,0:-1]))
y = Variable(torch.from_numpy(data[:TRAIN_SIZE:,-1]).long())-1
y_test = Variable(torch.from_numpy(data[TRAIN_SIZE:,-1]).long())-1


#Define Network Architecture
class Model(torch.nn.Module):

    def __init__(self):

        super().__init__()
        self.layer1 = torch.nn.Linear(21,17)
        self.layer2 = torch.nn.Linear(17,13)
        self.layer3 = torch.nn.Linear(13,9)
        self.layer4 = torch.nn.Linear(9,5)



    def forward(self,x):
        
        out1 = F.relu(self.layer1(x))
        out2 = F.relu(self.layer2(out1))
        out3 = F.relu(self.layer3(out2))
        y_pred = F.softmax(self.layer4(out3))

        return y_pred



model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


N_EPOCHS = 100000
for epoch in range(N_EPOCHS):
    optimizer.zero_grad() 
    y_pred = model(X)
    loss = criterion(y_pred,y)
    loss.backward()
    optimizer.step()
    print("Epoch",epoch+1,"Loss =",loss.data[0])


test_preds = model(X_test)
test_preds = torch.max(test_preds,1)[1]

correct = 0

for i in range(TEST_SIZE):
    if test_preds.data[i] == y_test.data[i]:
       correct += 1

print("Test Set Accuracy =",(correct/TEST_SIZE)*100)

torch.save(model.state_dict(),'trainedmodel')
