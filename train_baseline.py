from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

print("Veri seti yükleniyor...")
dataset = load_dataset("fancyzhx/ag_news")
train_data = dataset['train']
test_data = dataset['test']

X_train = train_data['text']
y_train = train_data['label']
X_test = test_data['text']
y_test = test_data['label']

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

print("Model eğitiliyor (bu birkaç dakika sürebilir)...")
pipeline.fit(X_train, y_train)

print("Test seti üzerinde doğruluk hesaplanıyor...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Task 1 - Baseline Test Accuracy: %{accuracy * 100:.2f}")

joblib.dump(pipeline, 'ag_news_baseline_model.pkl')
print("Model başarıyla 'ag_news_baseline_model.pkl' olarak kaydedildi.")