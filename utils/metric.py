import torch
import torch.nn.functional as F


def pytorch_acc(y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
    assert y_true.ndim == 1
    assert y_pred.ndim == 2
    num_classes = y_pred.size(1)

    y_pred = F.one_hot(y_pred.argmax(dim=1), num_classes)
    y_true = F.one_hot(y_true, num_classes)
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    accuracy = (tp + tn) / (tp + fp + tn + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)

    return accuracy.item(), precision.item(), recall.item(), f1.item()























