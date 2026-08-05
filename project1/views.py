from django.shortcuts import render
import pandas as pd
import io
import urllib, base64

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def index(request):
    context = {}

    if request.method == 'POST':
        try:
            csv_file = request.FILES['csv_file']
            
            df = pd.read_csv(csv_file)
            context['data_html'] = df.head().to_html(classes='data-table', index=False)
        
            model_type = request.POST.get('model_type')
            test_size_val = float(request.POST.get('test_size', 0.2))
        
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42)

            if model_type == 'knn':
                model = KNeighborsClassifier(n_neighbors=5) 
                model_name_display = "K-Nearest Neighbors"
            else:
                model = DecisionTreeClassifier(max_depth=5, random_state=42) 
                model_name_display = "Decision Tree"
            cols = df.columns

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            context['accuracy'] = round(acc * 100, 2) 
            context['selected_model'] = model_name_display
            context['test_size'] = test_size_val

            if len(cols) >= 2:
                plt.figure(figsize=(8, 5))
                plt.scatter(df[cols[0]], df[cols[1]], alpha=0.7, color='teal')
                plt.xlabel(cols[0])
                plt.ylabel(cols[1])
                plt.title(f"{cols[0]} vs {cols[1]}")
                plt.grid(True)

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                uri = 'data:image/png;base64,' + urllib.parse.quote(string)
                
                context['plot_uri'] = uri
                plt.close()
            else:
                context['error'] = "The dataset must contain at least 2 columns for a scatter plot."

        except Exception as e:
            context['error'] = f"Failed to process file: {str(e)}"

    return render(request, 'project1/index.html', context)