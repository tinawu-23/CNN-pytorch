import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.predict = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.hidden(x))
        prediction = self.predict(sig)
        return prediction


model = Model(input_size=2, hidden_size=3, output_size=1)
lossfunc = nn.MSELoss()
epochs = 1000
learnrate = 0.2
optimizer = torch.optim.SGD(model.parameters(), lr=learnrate)

x = [[0.5,0.9],[0.4,0.8],[0.3,0.6],[0.5,0.8],[0.1,0.4],[0.2,0.6],[0.8,0.8],[0.7,0.8],[0.2,0.5],[0.6, 0.9]]
y = [0.92, 0.91, 0.82, 0.95, 0.74, 0.75, 0.96, 0.94, 0.80, 0.91]

tensor_x = torch.FloatTensor(x)
tensor_y = torch.FloatTensor(y)
x, y = Variable(tensor_x), Variable(tensor_y)

for i in range(epochs):
    prediction = model(x)
    loss = lossfunc(prediction, y.view(-1,1))
    print("Epoch {}:\t Loss: {}".format(i+1, loss))
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

print('\n\n===================Training Data=====================\n')
prediction = model(x).detach().numpy()
for i, p in enumerate(prediction):
    print("Study hours {:.1f}; Sleep hours {:.1f}\t Real grade: {:.2f}\t Prediction: {:.2f}".format(
        x[i][0]*10, x[i][1]*10, y[i]*100, p[0]*100))

test = [[0.3, 0.9], [0.1, 0.3], [0.7, 0.7], [0.4, 0.5], [0.8, 0.9]]
tensor_test = torch.FloatTensor(test)
test = Variable(tensor_test)

print('\n\n===================Testing Data=====================\n')
pp = model(test).detach().numpy()
for i, p in enumerate(pp):
    print("Study hours {:.1f}; Sleep hours {:.1f}\t Prediction: {:.2f}".format(
        test[i][0]*10, test[i][1]*10, p[0]*100))
