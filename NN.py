import torch
import torch.nn as nn
import evotorch as et

# Define a simple network to evolve
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.layers(x)

# Fitness function: accuracy over a dataset
X = torch.randn(100, 20)
y = torch.randint(0, 2, (100,))

def fitness(solution):
    net = Net()
    et.nn.set_parameters(net, solution)  # load genome into model
    preds = net(X).argmax(dim=1)
    # a forloop of all the movements
    return (preds == y).float().mean().item()

# Define search space (genome length = number of parameters in Net)
dummy_net = Net()
num_params = sum(p.numel() for p in dummy_net.parameters())

problem = et.Problem(
    "max",         # maximize fitness
    fitness,
    solution_length=num_params
)

# Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
searcher = et.CMAES(problem, popsize=20)

# Run evolution
for gen in range(10):
    searcher.step()
    print(f"Generation {gen+1}, best fitness: {searcher.status['best_fitness']:.4f}")