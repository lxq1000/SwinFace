import torch
import torch.nn as nn

class AgeLoss(nn.Module):
    def __init__(self, total_iter, sigma=3.0):
        super().__init__()

        self.sigma = sigma
        self.total_iter = total_iter

    def forward(self, output, label, current_iter):
        output = output.to(torch.float32)
        label = label.to(torch.float32)

        la = current_iter / self.total_iter
        #print(la, current_iter, self.total_iter)
        dif_squ = (output - label) ** 2
        loss = (1 - la) / 2 * dif_squ + la * (1 - torch.exp(-dif_squ / (2 * self.sigma ** 2)))

        loss = torch.mean(loss)

        return loss