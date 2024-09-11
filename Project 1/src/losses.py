import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1
        
        y_true_scores = torch.gather(x_scores, 1, y.reshape(-1, 1))
        margins = self.delta + x_scores - y_true_scores
        margins[torch.arange(y.shape[0]), y] = 0
        hinge_losses = torch.sum(torch.max(torch.zeros_like(margins), margins), 1)
        loss = torch.mean(hinge_losses)
        
        self.grad_ctx['x'] = x
        self.grad_ctx['y'] = y
        self.grad_ctx['margins'] = margins

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        x, y, margins = self.grad_ctx['x'], self.grad_ctx['y'], self.grad_ctx['margins']
        
        G = (margins > 0).float()
        G[torch.arange(y.shape[0]), y] = -torch.sum(G, dim=1)
        
        grad = torch.matmul(x.t(), G) / y.shape[0]

        return grad
