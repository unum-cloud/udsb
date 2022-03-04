import torch


class ViaTorch:

    def __init__(self, side: int, backend=torch):
        self.backend = backend
        self.device = torch.device('cuda')
        # https://pytorch.org/docs/stable/generated/torch.rand.html
        self.mat = self.backend.rand(side, side, dtype=torch.float32).to(self.device)
        self.backend.cuda.synchronize(self.device)

    def moving_average(self):
        window: int = 3
        # https://stackoverflow.com/a/57897124
        ret = self.backend.cumsum(self.mat, dim=1, dtype=self.mat.dtype)
        ret[:, window:] = ret[:, window:] - ret[:, :-window]
        res = ret[:, window - 1:] / window
        self.backend.cuda.synchronize(self.device)
        return res

    def pearson_correlations(self):
        # https://pytorch.org/docs/stable/generated/torch.corrcoef.html
        res = self.backend.corrcoef(self.mat)
        self.backend.cuda.synchronize(self.device)
        return res

    def fft2d(self):
        # https://pytorch.org/docs/stable/generated/torch.fft.fft2.html
        res = self.backend.fft.fft2(self.mat)
        self.backend.cuda.synchronize(self.device)
        return res

    def matrix_multiply(self):
        # https://pytorch.org/docs/stable/generated/torch.matmul.html
        res = self.backend.matmul(self.mat, self.mat - self.backend.ones(self.mat.shape, dtype=self.mat.dtype).to(self.device))
        self.backend.cuda.synchronize(self.device)
        return res

    def singular_decomposition(self):
        # https://pytorch.org/docs/stable/generated/torch.linalg.svd.html
        res = self.backend.linalg.svd(self.mat)
        self.backend.cuda.synchronize(self.device)
        return res

    def flat_sort(self):
        # https://pytorch.org/docs/stable/generated/torch.sort.html
        res = self.backend.sort(self.mat)
        self.backend.cuda.synchronize(self.device)
        return res

    def flat_median(self):
        # https://pytorch.org/docs/stable/generated/torch.median.html
        res = self.backend.median(self.mat)
        self.backend.cuda.synchronize(self.device)
        return res

    def flat_sum(self):
        # https://pytorch.org/docs/stable/generated/torch.sum.html
        res = self.backend.sum(self.mat, dtype=self.mat.dtype)
        self.backend.cuda.synchronize(self.device)
        return res

    def close(self):
        pass
