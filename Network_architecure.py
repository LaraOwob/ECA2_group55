class NNController(nn.Module):
    def __init__(self, nq: int, nv: int, nu: int, hidden: int = 24, freq_hz: float = 1.5):
        super().__init__()
        self.nq, self.nv, self.nu = nq, nv, nu
        self.in_dim = nq + nv + 2 + 2  # correct
        self.fc1 = nn.Linear(self.in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, nu)  # output matches number of actuators
        self.act = nn.Tanh()
        self.freq_hz = freq_hz
        self.u_prev = None

    def forward(self, qpos, qvel, t, target_pos, current_pos):
        w = 2 * torch.pi * self.freq_hz
        tt = torch.tensor(t, dtype=torch.float32)
        time_feats = torch.stack([torch.sin(w * tt), torch.cos(w * tt)])
        delta = target_pos[:2] - current_pos[:2]
        obs = torch.cat([qpos, qvel, time_feats, delta], dim=0)
        x = self.act(self.fc1(obs))
        x = self.act(self.fc2(x))
        y = self.act(self.fc3(x))
        return y  # in [-1,1]
    

def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    # Normally, you would use the genes of an individual as the weights,
    # Here we set them randomly for simplicity.
    w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs
    return outputs * np.pi