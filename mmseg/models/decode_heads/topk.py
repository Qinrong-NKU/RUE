import torch
from torch import nn


def conv3x3(in_chs: int, out_chs: int = 16) -> nn.Conv2d:
    return nn.Conv2d(
        in_chs,
        out_chs,
        kernel_size=3,
        stride=2,
        padding=1
    )

def conv1x1(in_chs: int, out_chs: int = 16) -> nn.Conv2d:
    return nn.Conv2d(
        in_chs,
        out_chs,
        kernel_size=1,
        stride=1,
        padding=0
    )

class MapBlock(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int = 16,
    ) -> None:
        super(MapBlock, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(in_chs, out_chs),
            nn.BatchNorm2d(out_chs),
            nn.LeakyReLU(0.2),
            conv3x3(out_chs, out_chs),
            # nn.BatchNorm2d(out_chs)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        return out
    
class PerturbedTopK(nn.Module):

    def __init__(self, k: int, num_samples: int = 500, sigma: float = 0.05):
        super(PerturbedTopK, self).__init__()

        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def __call__(self, x):
        # Return the output of the PerturbedTopKFunction, applied to the input tensor
        # using the k, num_samples, and sigma attributes as arguments
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)


class PerturbedTopKFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 500, sigma: float = 0.05):
        b, d = x.shape

        # Generate Gaussian noise with specified number of samples and standard deviation
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)

        # Add noise to the input tensor
        perturbed_x = x[:, None, :] + noise * sigma  # b, nS, d

        # Perform top-k pooling on the perturbed tensor
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)

        # Get the indices of the top k elements
        indices = topk_results.indices  # b, nS, k

        # Sort the indices in ascending order
        indices = torch.sort(indices, dim=-1).values  # b, nS, k

        # Convert the indices to one-hot tensors
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()

        # Average the one-hot tensors to get the final output
        indicators = perturbed_output.mean(dim=1)  # b, k, d

        # Save constants and tensors for backward pass
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        # If there is no gradient to backpropagate, return tuple of None values
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise

        # Calculate expected gradient
        expected_gradient = (
                                    torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, noise_gradient)
                                    / ctx.num_samples
                                    / ctx.sigma
                            ) * float(ctx.k)

        grad_input = torch.einsum("bkd,bkde->be", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)


class Selector(nn.Module):


    def __init__(self, high_size=(4000,4000), score_size=(32,32), k=10, num_samples=1024, sigma=0.05, patch_size=128
                 ):
        super().__init__()

        self.patch_size = patch_size
        self.k = k


        self.scorer = MapBlock(in_chs=512,out_chs=1)

        self.TOPK = PerturbedTopK(k, num_samples, sigma)

        h, w = high_size
        self.h_score, self.w_score = score_size

        self.scale_h = h // self.h_score
        self.scale_w = w // self.w_score
        padded_h = self.scale_h * self.h_score + patch_size - 1
        padded_w = self.scale_w * self.w_score + patch_size - 1
        top_pad = (patch_size - self.scale_h) // 2
        left_pad = (patch_size - self.scale_w) // 2
        bottom_pad = padded_h - top_pad - h
        right_pad = padded_w - left_pad - w

        self.padding = (left_pad, right_pad, top_pad, bottom_pad)
        



    def forward(self, x_high, x_low):

        b, c = x_high.shape[:2]
        patch_size = self.patch_size
        device = x_high.device

        ### Score patches to get indicators
        scores_2d = self.scorer(x_low)
        print(scores_2d.shape)
        scores_2d=scores_2d.squeeze(1)
        scores_1d = scores_2d.view(b, -1)
        # # 0 -1 normalization
        # scores_min = scores_1d.min(axis=-1, keepdims=True)[0]
        # scores_max = scores_1d.max(axis=-1, keepdims=True)[0]
        # scores_1d = (scores_1d - scores_min) / (scores_max - scores_min + 1e-5)
        # 
        # indicators = self.TOPK(scores_1d).view(b, self.k, self.h_score, self.w_score)
        # 
        # # Pad image      
        # x_high_pad = torch.nn.functional.pad(x_high, self.padding, "constant", 0)
        # 
        # # Extract patches
        # patches = torch.zeros((b, self.k, c, patch_size, patch_size)).to(device)
        # for i in range(self.h_score):
        #     for j in range(self.w_score):
        #         start_h = i * self.scale_h
        #         start_w = j * self.scale_w
        # 
        #         # (b, c, patch_size, patch_size)
        #         current_patches = x_high_pad[:, :, start_h: start_h + patch_size, start_w: start_w + patch_size]
        #         weight = indicators[:, :, i, j]  # b, k
        # 
        #         # Broacast, element-wise mult.
        #         patches += torch.einsum('bchw,bk->bkchw', current_patches, weight) 
        # return patches
