from torchsummary import summary
import torch

from ResNet import DKU_ResNet

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use", device)
    model = DKU_ResNet(num_classes=2).to(device)
    summary(model,(1,513,500))
