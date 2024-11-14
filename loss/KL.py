import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import KLDivLoss

class WxKLDivLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, pred, label):
        """_summary_: forward kl divergence D(pred || label) = pred * log(pred / label)

        Args:
            pred (_type_): _description_
            label (_type_): _description_
        """
        return (label * (torch.log(label) - pred)).sum(dim=-1).mean()

def test():
    shape = [5, 10]
    pred, label = torch.rand(shape), torch.rand(shape)
    print("diff of div", KLDivLoss(reduction="batchmean")(pred, label), WxKLDivLoss()(pred, label))

if __name__ == "__main__":
    test()