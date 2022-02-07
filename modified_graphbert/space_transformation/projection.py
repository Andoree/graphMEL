import logging
import torch
from torch import nn


class ProjectionLayer(nn.Module):
    def __init__(self, emb_size=300, hidden_sizes=(1024, 512), target_size=300, device='cuda'):
        super().__init__()
        self.device = device
        self.layers = nn.Sequential(
            nn.Linear(emb_size, hidden_sizes[0]),
            nn.ELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ELU(),
            nn.Linear(hidden_sizes[1], target_size)
        ).to(device).train()

    def forward(self, x):
        return self.layers(x)

    def train_and_evaluate(self, train_loader, dev_loader, loss_function, optimizer, scheduler, epochs):
        for epoch in range(epochs):
            logging.info(f'Starting training epoch {epoch + 1}')
            losses = []
            self.train()
            for i, data in enumerate(train_loader):
                anchor, positive, negative = data
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

                optimizer.zero_grad()
                outputs = self(anchor)
                loss = loss_function(outputs, positive, target=torch.Tensor((1,)).to(self.device)) + \
                    loss_function(outputs, negative, target=torch.Tensor((-1,)).to(self.device))

                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if i % 200 == 199:
                    logging.info('Loss after mini-batch %5d: %.10f' % (i + 1, sum(losses) / len(losses)))
                    losses = []

            logging.info(f'Starting evaluation epoch {epoch + 1}')
            self.eval()
            dev_losses = []
            with torch.no_grad():
                for i, data in enumerate(dev_loader):
                    anchor, positive, negative = data
                    anchor, positive, negative = anchor.to(self.device), positive.to(self.device), \
                                                 negative.to(self.device)
                    out = self(anchor)
                    loss = loss_function(out, positive, target=torch.Tensor((1,)).to(self.device)) + \
                        loss_function(out, negative, target=torch.Tensor((-1,)).to(self.device))
                    dev_losses.append(loss.item())

            logging.info('Dev Loss %.10f' % (sum(dev_losses) / len(dev_losses)))
            scheduler.step()

    def save(self, path):
        torch.save(self.state_dict(), path)
