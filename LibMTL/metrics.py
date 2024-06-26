import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AbsMetric(object):
    r"""An abstract class for the performance metrics of a task. 

    Attributes:
        record (list): A list of the metric scores in every iteration.
        bs (list): A list of the number of data in every iteration.
    """
    def __init__(self):
        self.record = []
        self.bs = []
    
    @property
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        pass
    
    @property
    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).

        Return:
            list: A list of metric scores.
        """
        pass
    
    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends).
        """
        self.record = []
        self.bs = []
    
# accuracy
class AccMetric(AbsMetric):
    r"""Calculate the accuracy.
    """
    def __init__(self):
        super(AccMetric, self).__init__()
        
    def update_fun(self, pred, gt):
        r"""
        """
        # TODO: Update code in order to restore AccMetric to OG
        # pred = F.softmax(pred, dim=-1).max(-1)[1]
        pred = torch.nn.functional.one_hot(torch.argmax(pred, dim = 1), num_classes=pred.shape[1])
        gt = gt.squeeze(1)
        gt = gt.squeeze(1).float()
        pred_labels = torch.argmax(pred, dim=1)
        gt_labels = torch.argmax(gt, dim=1)
        accuracy = (pred_labels == gt_labels).float().mean().item()

        self.record.append(accuracy)
        self.bs.append(pred.size()[0])
        
    def score_fun(self):
        r"""
        """
        return [sum(r * b for r, b in zip(self.record, self.bs)) / sum(self.bs)]


# L1 Error
class L1Metric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """
    def __init__(self):
        super(L1Metric, self).__init__()
        
    def update_fun(self, pred, gt):
        r"""
        """
        abs_err = torch.abs(pred - gt)
        self.record.append(abs_err.item())
        self.bs.append(pred.size()[0])
        
    def score_fun(self):
        r"""
        """
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return [(records*batch_size).sum()/(sum(batch_size))]
