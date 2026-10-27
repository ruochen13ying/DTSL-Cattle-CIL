import numpy as np


def last_accuracy(stage_accuracies):
    return float(stage_accuracies[-1])


def average_incremental_accuracy(stage_accuracies):
    return float(np.mean(stage_accuracies))


def backward_transfer(accuracy_matrix):
    '''
    accuracy_matrix[t][j] = accuracy on task j after learning stage t.
    BWT = mean_j (R[T,j] - R[j,j]) over all tasks before the final task.
    '''
    total_stages = len(accuracy_matrix)
    if total_stages <= 1:
        return 0.0
    final_row = accuracy_matrix[-1]
    values = [
        final_row[j] - accuracy_matrix[j][j]
        for j in range(total_stages - 1)
    ]
    return float(np.mean(values))
