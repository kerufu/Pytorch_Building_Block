import torch
import kornia

class MultiScaleDerivativeLoss:
    def __init__(self, p, M, is_scharr=True, device="cuda", reduction="mean"):
        # p: error norm
        # M: scale
        # reduction:
        #   mean: average over batch
        #   sample: the loss for each sample

        if is_scharr:
            self.operator = kornia.filters.Sobel()
        else:
            self.operator = kornia.filters.Laplacian(5)

        self.p = p
        self.M = M
        self.device = device
        self.reduction = reduction

    def rescale(self, predict, ground_truth):
        predict = kornia.geometry.transform.rescale(predict, (0.5, 0.5))
        ground_truth = kornia.geometry.transform.rescale(ground_truth, (0.5, 0.5))
        return predict, ground_truth

    def gradient_difference(self, predict, ground_truth):
        difference = self.operator(predict) - self.operator(ground_truth)
        difference = torch.abs(difference) ** self.p
        if self.reduction == "mean":
            difference = torch.mean(difference)
        else:
            difference = torch.mean(difference, list(range(1, len(difference.shape))))
        return difference

    def __call__(self, predict, ground_truth):
        rescaled_predict, rescaled_ground_truth = torch.clone(predict), torch.clone(ground_truth)

        result = 0
        for _ in range(self.M):
            difference = self.gradient_difference(rescaled_predict, rescaled_ground_truth)
            result += difference
            rescaled_predict, rescaled_ground_truth = self.rescale(rescaled_predict, rescaled_ground_truth)

        result = result / self.M

        return result


