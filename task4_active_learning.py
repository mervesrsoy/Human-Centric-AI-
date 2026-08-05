from datasets import load_dataset
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print("Loading data and Baseline Model...")
dataset = load_dataset("fancyzhx/ag_news")

pool_texts = dataset['train']['text'][:5000]
pool_labels = np.array(dataset['train']['label'][:5000])

X_test = dataset['test']['text']
y_test = np.array(dataset['test']['label'])

baseline_model = joblib.load('ag_news_baseline_model.pkl')

def simulated_expert(true_labels):
    preds = []
    for label in true_labels:
        if label in [1, 3]:
            preds.append(label)
        else:
            preds.append(np.random.choice([0, 1, 2, 3]))
    return np.array(preds)

print("Step 1: Active Learning (Uncertainty Sampling)...")

pool_probs = baseline_model.predict_proba(pool_texts)
pool_preds = np.argmax(pool_probs, axis=1)
pool_confidence = np.max(pool_probs, axis=1)

n_queries = 500
uncertain_indices = np.argsort(pool_confidence)[:n_queries]

queried_texts = [pool_texts[i] for i in uncertain_indices]
queried_labels = pool_labels[uncertain_indices]
queried_model_preds = pool_preds[uncertain_indices]

print(f"Step 2: Querying the expert for {n_queries} uncertain samples...")
queried_expert_preds = simulated_expert(queried_labels)

print("Step 3: Training the Allocator (Meta-Model)...")

allocator_targets = []
for i in range(n_queries):
    expert_correct = (queried_expert_preds[i] == queried_labels[i])
    model_correct = (queried_model_preds[i] == queried_labels[i])
    
    if expert_correct and not model_correct:
        allocator_targets.append(1)
    else:
        allocator_targets.append(0)

allocator = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('classifier', LogisticRegression(random_state=42))
])
allocator.fit(queried_texts, allocator_targets)

print("Step 4: Evaluating the Dynamic Team on Test Set...")

test_model_preds = baseline_model.predict(X_test)
test_expert_preds = simulated_expert(y_test)

defer_decisions = allocator.predict(X_test)

team_preds = []
deferred_count = 0

for i in range(len(y_test)):
    if defer_decisions[i] == 1:
        team_preds.append(test_expert_preds[i])
        deferred_count += 1
    else:
        team_preds.append(test_model_preds[i])

team_accuracy = accuracy_score(y_test, team_preds)
deferral_rate = (deferred_count / len(y_test)) * 100

print("-" * 40)
print(f"Task 4 - Active Learning Team Accuracy: {team_accuracy * 100:.2f}%")
print(f"Dynamic Deferral Rate: {deferral_rate:.2f}%")
print("-" * 40)