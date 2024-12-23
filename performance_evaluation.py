import torch

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), correct / total

def compare_models(original_model, quantized_model, dataloader, criterion, device):
    original_loss, original_accuracy = evaluate_model(original_model, dataloader, criterion, device)
    quantized_loss, quantized_accuracy = evaluate_model(quantized_model, dataloader, criterion, device)
    print(f'Original Model - Loss: {original_loss}, Accuracy: {original_accuracy}')
    print(f'Quantized Model - Loss: {quantized_loss}, Accuracy: {quantized_accuracy}')