from django.shortcuts import render
from palmerpenguins import load_penguins
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#dcd6e8',
    'axes.labelcolor': '#3a3042',
    'xtick.color': '#80758a',
    'ytick.color': '#80758a',
    'text.color': '#3a3042',
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff'
})

def index(request):
    context = {}

    lambda_val = 0.0
    model_type = 'tree'
    selected_feature = 'bill_length_mm'
    
    if request.method == 'POST':
        lambda_val = float(request.POST.get('lambda_val', 0.0))
        model_type = request.POST.get('model_type', 'tree')
        selected_feature = request.POST.get('selected_feature', 'bill_length_mm')

    context['lambda_val'] = lambda_val
    context['model_type'] = model_type
    context['selected_feature'] = selected_feature

    # cleaning data
    penguins = load_penguins()
    penguins = penguins.dropna()
    y = penguins['species']
    X = penguins.drop(columns=['species'])
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_acc = 0
    best_complexity_val = 0
    best_score = float('inf')
    best_model = None
    scaler = StandardScaler()

    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
      #model training
    if model_type == 'tree':
        leaf_nodes_options = [2, 3, 4, 5, 7, 10, None] 
        for max_leaves in leaf_nodes_options:
            clf = DecisionTreeClassifier(max_leaf_nodes=max_leaves, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            leaves = clf.get_n_leaves()

            score = (1.0 - acc) + (lambda_val * leaves)

            if score < best_score:
                best_score = score
                best_model = clf
                best_acc = acc
                best_complexity_val = leaves

        plot_tree(best_model, feature_names=X.columns, class_names=best_model.classes_, filled=True, rounded=True, fontsize=10, ax=ax)
        context['complexity_label'] = "Number of Leaves (Tree Complexity)"

    elif model_type == 'logistic':
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        c_options = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        for c_val in c_options:
            clf = LogisticRegression(penalty='l1', C=c_val, solver='saga', random_state=42, max_iter=5000)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            
            l1_norm = np.sum(np.abs(clf.coef_))
            score = (1.0 - acc) + (lambda_val * l1_norm)

            if score < best_score:
                best_score = score
                best_model = clf
                best_acc = acc
                best_complexity_val = round(l1_norm, 2)

        importance = np.sum(np.abs(best_model.coef_), axis=0)
        
        bars = ax.bar(X.columns, importance, color='#a288e3', alpha=0.85, edgecolor='#8860d0', linewidth=1)
        ax.set_xticks(range(len(X.columns)))
        ax.set_xticklabels(X.columns, rotation=45, ha='right')
        ax.set_title('Logistic Regression - Feature Importances (L1 Norm)', pad=15, fontweight='bold', color='#3a3042')
        ax.set_ylabel('Weight Magnitude', fontweight='500')
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#dcd6e8')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        context['complexity_label'] = "L1 Norm of Weights (Sparsity)"

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    context['model_plot'] = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    context['accuracy'] = round(best_acc * 100, 2)
    context['complexity_val'] = best_complexity_val
       #task4
    if request.method == 'POST' and request.POST.get('run_cf'):
        cf_index = int(request.POST.get('cf_index', 0))
        cf_target = request.POST.get('cf_target')
        context['cf_requested'] = True
        context['cf_target'] = cf_target
        context['cf_index'] = cf_index
        
        if cf_index >= len(X_test): cf_index = 0
            
        x_orig = X_test.iloc[cf_index]
        x_orig_df = pd.DataFrame([x_orig])
        
        if model_type == 'logistic':
            orig_pred = best_model.predict(scaler.transform(x_orig_df))[0]
        else:
            orig_pred = best_model.predict(x_orig_df)[0]
            
        context['orig_pred'] = orig_pred

        if orig_pred == cf_target:
            context['cf_found'] = False
            context['orig_pred'] += " (Already matches target!)"
        else:
            N = 2000
            X_fake = pd.DataFrame([x_orig.to_dict() for _ in range(N)])
            
            for col in X.columns:
                if X_train[col].nunique() > 2: 
                    X_fake[col] += np.random.normal(0, X_train[col].std(), N)
                else: 
                    flip_mask = np.random.rand(N) > 0.8
                    X_fake[col] = np.where(flip_mask, 1 - X_fake[col], X_fake[col])

            if model_type == 'logistic':
                preds = best_model.predict(scaler.transform(X_fake))
            else:
                preds = best_model.predict(X_fake)

            valid_idx = np.where(preds == cf_target)[0]
            
            if len(valid_idx) > 0:
                X_valid = X_fake.iloc[valid_idx]
                mad = (X_train - X_train.median()).abs().median().replace(0, 1e-6)
                distances = np.sum(np.abs(X_valid - x_orig) / mad, axis=1)
                best_cf = X_valid.iloc[distances.argmin()]
                
                comparison = [{'feature': col, 'orig': x_orig[col], 'cf': best_cf[col], 'diff': best_cf[col] - x_orig[col]} 
                              for col in X.columns if abs(best_cf[col] - x_orig[col]) > 0.05]
                
                context['cf_comparison'] = comparison
                context['cf_found'] = True
            else:
                context['cf_found'] = False

    # task5
    if request.method == 'POST' and request.POST.get('run_plots'):
        classes = best_model.classes_
        colors = ['#a288e3', '#ffb7b2', '#95d9c3'] 
        
        #calculations
        grid_pdp = np.linspace(X_train[selected_feature].min(), X_train[selected_feature].max(), 50)
        pdp_results = np.zeros((len(grid_pdp), len(classes)))
        
        for i, val in enumerate(grid_pdp):
            X_temp = X_train.copy()
            X_temp[selected_feature] = val 
            if model_type == 'logistic':
                probs = best_model.predict_proba(scaler.transform(X_temp))
            else:
                probs = best_model.predict_proba(X_temp)
            pdp_results[i, :] = np.mean(probs, axis=0) 

        fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=120)
        for c_idx, c_name in enumerate(classes):
            ax1.plot(grid_pdp, pdp_results[:, c_idx], label=c_name, color=colors[c_idx], linewidth=2.5)
        
        ax1.set_title(f'PDP: {selected_feature}', fontweight='bold', pad=10)
        ax1.set_xlabel(selected_feature, fontweight='500')
        ax1.set_ylabel('Average Probability', fontweight='500')
        ax1.legend(frameon=False)
        ax1.grid(axis='y', linestyle='--', alpha=0.4, color='#dcd6e8')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        context['pdp_plot'] = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig1)

        #ale
        quantiles = np.unique(np.percentile(X_train[selected_feature], np.linspace(0, 100, 16)))
        centers = (quantiles[:-1] + quantiles[1:]) / 2
        local_effects = np.zeros((len(centers), len(classes)))

        for k in range(len(quantiles)-1):
            z_low, z_high = quantiles[k], quantiles[k+1]
            idx = (X_train[selected_feature] >= z_low) & (X_train[selected_feature] <= z_high)
            X_bin = X_train[idx].copy()
            
            if len(X_bin) == 0: continue
            
            X_high, X_low = X_bin.copy(), X_bin.copy()
            X_high[selected_feature] = z_high
            X_low[selected_feature] = z_low
            
            if model_type == 'logistic':
                p_high = best_model.predict_proba(scaler.transform(X_high))
                p_low = best_model.predict_proba(scaler.transform(X_low))
            else:
                p_high = best_model.predict_proba(X_high)
                p_low = best_model.predict_proba(X_low)
                
            local_effects[k, :] = np.mean(p_high - p_low, axis=0)

        ale_accum = np.cumsum(local_effects, axis=0)
        ale_accum = ale_accum - np.mean(ale_accum, axis=0)

        fig2, ax2 = plt.subplots(figsize=(6, 4.5), dpi=120)
        for c_idx, c_name in enumerate(classes):
            ax2.plot(centers, ale_accum[:, c_idx], label=c_name, color=colors[c_idx], linewidth=2.5, marker='o', markersize=5)
        
        ax2.set_title(f'ALE: {selected_feature}', fontweight='bold', pad=10)
        ax2.set_xlabel(selected_feature, fontweight='500')
        ax2.set_ylabel('Accumulated Local Effect', fontweight='500')
        ax2.axhline(0, color='#dcd6e8', linestyle='--', linewidth=1.5)
        ax2.legend(frameon=False)
        ax2.grid(axis='y', linestyle='--', alpha=0.4, color='#dcd6e8')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        context['ale_plot'] = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig2)

    return render(request, 'project2/index.html', context)