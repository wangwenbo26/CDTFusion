import torch
import torch.nn as nn

class TaskInteraction(nn.Module):
    def __init__(self, in_channels):
        super(TaskInteraction, self).__init__()

        self.concat = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.bn = nn.BatchNorm2d(in_channels)

        self.self_attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.in_channels = in_channels

        self.bn_x1 = nn.BatchNorm2d(in_channels)
        self.bn_x2 = nn.BatchNorm2d(in_channels)

        self.linear_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.linear_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.linear_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x1, x2):

        x = torch.cat((x1, x2), dim=1)
        x = x1 + self.concat(x)

        x = self.bn(x)

        x = x.view(x.shape[0], x.shape[1], -1).permute(2, 0, 1)
        x, _ = self.self_attention(x, x, x)
        x = x.permute(1, 2, 0).view(x1.shape)

        x = self.mlp(x)
        x1 = self.bn_x1(x1)
        x2 = self.bn_x2(x)

        q = self.linear_q(x1)
        k = self.linear_k(x2)
        v = self.linear_v(x2)

        q = q.view(q.size(0), q.size(1), -1)
        k = k.view(k.size(0), k.size(1), -1)
        v = v.view(v.size(0), v.size(1), -1)

        attention_scores = torch.bmm(q.transpose(1, 2), k)

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=2)

        weighted_sum = torch.bmm(attention_weights, v.transpose(1, 2))

        weighted_sum = weighted_sum.view(weighted_sum.size(0), self.in_channels, x2.size(2), x2.size(3))

        output = x1 + weighted_sum

        return output



class CatAndConvolve(nn.Module):
    def __init__(self):
        super(CatAndConvolve, self).__init__()
        self.conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)

    def forward(self, tensor1, tensor2):

        concatenated_tensor = torch.cat((tensor1, tensor2), dim=1)

        output_tensor = self.conv(concatenated_tensor)

        return output_tensor