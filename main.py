from torch.utils.data import DataLoader
from quantization_utils import apply_kmeans_quantization, quantization_aware_training
from performance_evaluation import compare_models
import copy  # 원본 모델 복사를 위해 추가

def main():
    # Data preparation
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # Model loading
    model = models.resnet18(pretrained=True)
    model.cuda()

    # Quantization Aware Training (QAT)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    quantization_aware_training(model, trainloader, criterion, optimizer)

    # Apply quantization
    original_model = copy.deepcopy(model)  # 원본 모델 복사
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = apply_kmeans_quantization(param.data)  # 양자화 적용

    # Compare original and quantized models
    compare_models(original_model, model, trainloader, criterion, 'cuda')

if __name__ == '__main__':
    main()