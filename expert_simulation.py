from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

print("Veri seti test için yükleniyor...")
dataset = load_dataset("fancyzhx/ag_news")
y_test = np.array(dataset['test']['label'])

def simulated_expert(true_labels):
    
    expert_preds = []
    for label in true_labels:
        if label in [1, 3]:
            expert_preds.append(label)
        else:
            expert_preds.append(np.random.choice([0, 1, 2, 3]))
    return np.array(expert_preds)

print("Uzman simülasyonu çalışıyor...")
expert_predictions = simulated_expert(y_test)

expert_accuracy = accuracy_score(y_test, expert_predictions)
print(f"Task 2 - Simulated Expert Test Accuracy: %{expert_accuracy * 100:.2f}")