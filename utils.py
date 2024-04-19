import torch
import torch.nn.functional as F
import numpy as np

def getTestAcc(classifier, vectorModel, test_loader, device, batch_size):

    test_acc = []
    classifier.eval()

    for inputs_test, targets_test in test_loader:

        inputs_test = inputs_test.to(device)
        targets_test = targets_test.to(device)

        output_test = classifier(vectorModel(inputs_test))

        y_hat_test = torch.argmax(output_test, 1)
        score_test = torch.eq(y_hat_test, targets_test).sum()
        test_acc.append((score_test.item()/batch_size)*100)

    return np.mean(np.array(test_acc))

def get_bottom_indices(values, prune_limit):

    sorted_indices = sorted(range(len(values)), key = lambda k : values[k])
    non_pruned_indices = sorted_indices[prune_limit : ]
    # non_pruned_values = [values[i] for i in non_pruned_indices]

    return non_pruned_indices

def findEntropy(matrix):
    flattern_tensor = matrix.view(-1)
    probabilities = F.softmax(flattern_tensor, dim=0)
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy.item()