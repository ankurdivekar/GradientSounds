import pyaudio
import numpy as np
import wave
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Code modified from: http://blog.christianperone.com/2019/08/listening-to-the-neural-network-gradient-norms-during-training/

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.ordered_layers = [self.conv1,
                               self.conv2,
                               self.fc1,
                               self.fc2]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def open_stream(fs):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    return p, stream


def generate_tone(fs, freq, duration):
    npsin = np.sin(2 * np.pi * np.arange(fs*duration) * freq / fs)
    samples = npsin.astype(np.float32)
    return 0.1 * samples


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    fs = 44100
    duration = 0.01
    f = 200.0
    p, stream = open_stream(fs)
    frames = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        norms = []
        for layer in model.ordered_layers:
            norm_grad = layer.weight.grad.norm()
            norms.append(norm_grad)
            tone = f + ((norm_grad.numpy()) * 100.0)
            tone = tone.astype(np.float32)
            samples = generate_tone(fs, tone, duration)
            frames.append(samples)
        silence = np.zeros(samples.shape[0] * 2,
                           dtype=np.float32)
        frames.append(silence)
        optimizer.step()
        # Just 200 steps per epoach
        if batch_idx == 200:
            break

    wf = wave.open("sgd_lr_1_0_bs256.wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()


def run_main():
    device = torch.device("cpu")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=256, shuffle=True)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(1, 5):
        train(model, device, train_loader, optimizer, epoch)


if __name__ == "__main__":
    run_main()
