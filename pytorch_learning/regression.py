import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


x = torch.linspace(-1, 1, 100).reshape(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())

# plt.scatter(x, y)
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.predict(x)

        return x

net = Net(1, 10, 1)
# print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

plt.ion()   # something about plotting

for t in range(200):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 12, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
