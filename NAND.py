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


model = Model(input_size=2, hidden_size=2, output_size=1)
lossfunc = nn.MSELoss()
epochs = 1000
learnrate = 0.2
optimizer = torch.optim.SGD(model.parameters(), lr=learnrate)

x = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
y = [0,1,1,1]

tensor_x = torch.FloatTensor(x)
tensor_y = torch.FloatTensor(y)
x, y = Variable(tensor_x), Variable(tensor_y)

for i in range(epochs):
    prediction = model(x)
    loss = lossfunc(prediction, y.view(-1, 1))
    print("Epoch {}:\t Loss: {}".format(i+1, loss))
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

test = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
tensor_test = torch.FloatTensor(test)
test = Variable(tensor_test)

print('\n\n========Result=======\n')

prediction = model(test).detach().numpy()
res = []
for p in prediction:
    if p[0] < 0.5:
        res.append(0)
    else:
        res.append(1)
print('A\t B\t A NAND B\n')
for i in range(len(res)):
    print("{}\t {}\t     {}".format(int(test[i][0]), int(test[i][1]), res[i]))
