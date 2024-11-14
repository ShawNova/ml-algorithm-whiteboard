import torch
from torch import nn
from torch.nn import BCELoss, CrossEntropyLoss
import torch.nn.functional as F

class WxBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction # mean or sum

    def forward(self, y_pred, y_label):
        """
        Function: bce loss without sigmoid: L = - y_label * log(y_pred) - (1 - y_label) * log(1 - y_pred)

        Args:
        y_pred: (bs, 1)
        y_label: (bs, 1) why 1? -> dataloader provides the dimension

        """
        # y_pred = F.sigmoid(y_pred)
        loss = - y_label * torch.log(y_pred) - (1 - y_label) * torch.log(1 - y_pred)
        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()

        return loss

class WxCrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, label):
        """_summary_: math: L = sum of i for - y_label_i * log(y_pred_i)

        Args:
            pred (): (bs, n)
            label (): (bs, n)
        """
        pred = F.softmax(pred, dim=-1)
        return (- label * torch.log(pred)).sum(dim=-1).mean()


def test_ce_loss():
    shape = [10, 1]
    y_pred = torch.rand(shape)
    y_label = torch.rand(shape)
    bce_loss_official = BCELoss()
    bce_loss_diy = WxBinaryCrossEntropyLoss()
    print("diff of bce loss:", bce_loss_official(y_pred, y_label), bce_loss_diy(y_pred, y_label))

    shape = [10, 5]
    y_pred = torch.rand(shape)
    y_label = F.softmax(torch.rand(shape), dim=-1)
    ce_official = CrossEntropyLoss(reduction='mean')
    ce_diy = WxCrossEntropyLoss()
    print("diff of ce loss:", ce_official(y_pred, y_label), ce_diy(y_pred, y_label))


if __name__ == "__main__":
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    test_ce_loss()
