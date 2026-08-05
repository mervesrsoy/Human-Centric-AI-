from datasets import load_dataset
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

print("Loading data and Task 1 Model...")
dataset = load_dataset("fancyzhx/ag_news")
X_test = dataset['test']['text']
y_test = np.array(dataset['test']['label'])

pipeline = joblib.load('ag_news_baseline_model.pkl')

def simulated_expert(true_label):
    if true_label in [1, 3]: #sport and tech
        return true_label
    else:
        return np.random.choice([0, 1, 2, 3])

print("Applying Learning-to-Defer strategy...")

y_prob = pipeline.predict_proba(X_test)
model_confidence = np.max(y_prob, axis=1)
model_predictions = np.argmax(y_prob, axis=1)

CONFIDENCE_THRESHOLD = 0.85

team_predictions = []
deferred_count = 0

for i in range(len(y_test)):
    if model_confidence[i] < CONFIDENCE_THRESHOLD:
        expert_pred = simulated_expert(y_test[i])
        team_predictions.append(expert_pred)
        deferred_count += 1
    else:
        team_predictions.append(model_predictions[i])

team_accuracy = accuracy_score(y_test, team_predictions)
deferral_rate = (deferred_count / len(y_test)) * 100

print("-" * 40)
print(f"Task 3 - Team (Model + Expert) Accuracy: {team_accuracy * 100:.2f}%")
print(f"Deferral Rate: {deferral_rate:.2f}%")
print("-" * 40)