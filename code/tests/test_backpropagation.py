import torch


class SimpleModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 2)
        self.linear_1.weight = torch.nn.parameter.Parameter(torch.tensor([[1, 1], [1, -1]], dtype=torch.float32))
        self.linear_1.bias = torch.nn.parameter.Parameter(torch.tensor([0, 0], dtype=torch.float32))
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(2, 1)
        self.linear_2.weight = torch.nn.parameter.Parameter(torch.tensor([1, 1], dtype=torch.float32))
        self.linear_2.bias = torch.nn.parameter.Parameter(torch.tensor([-0.5], dtype=torch.float32))

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x


class LargerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 2)
        self.linear_1.weight = torch.nn.parameter.Parameter(torch.tensor([[1, 1], [1, -1]], dtype=torch.float32))
        self.linear_1.bias = torch.nn.parameter.Parameter(torch.tensor([0, 0], dtype=torch.float32))
        self.relu1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(2, 2)
        self.linear_2.weight = torch.nn.parameter.Parameter(torch.tensor([[1, 1], [1, -1]], dtype=torch.float32))
        self.linear_2.bias = torch.nn.parameter.Parameter(torch.tensor([-0.5, 0], dtype=torch.float32))
        self.relu2 = torch.nn.ReLU()
        self.linear_3 = torch.nn.Linear(2, 2)
        self.linear_3.weight = torch.nn.parameter.Parameter(torch.tensor([[-1, 1], [0, 1]], dtype=torch.float32))
        self.linear_3.bias = torch.nn.parameter.Parameter(torch.tensor([3, 0], dtype=torch.float32))

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu2(x)
        x = self.linear_3(x)
        return x
