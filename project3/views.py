from django.shortcuts import render
from datasets import load_dataset
import random
from django.shortcuts import redirect

def index(request):
    context = {
        'baseline_accuracy': 91.21,
        'expert_accuracy': 62.26,
        'team_accuracy': 86.61,
        'deferral_rate': 28.86,
        'al_team_accuracy': 91.21,
        'al_deferral_rate': 0.00,
    }
    return render(request, 'project3/index.html', context)

try:
    demo_dataset = load_dataset("fancyzhx/ag_news", split="train[:200]")
    pool_texts = demo_dataset['text']
except:
    pool_texts = ["Error loading dataset. This is a placeholder text."]

def active_labeling(request):
    if 'human_labels' not in request.session:
        request.session['human_labels'] = 0

    if request.method == 'POST':
        request.session['human_labels'] += 1
        return redirect('project3:active_labeling')

    context = {
        'text_to_label': random.choice(pool_texts),
        'labeled_count': request.session['human_labels']
    }
    return render(request, 'project3/label.html', context)