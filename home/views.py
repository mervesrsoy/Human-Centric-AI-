# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render

def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Merve Sarısoy", "matriculation": "674397"},
    ]
    
    projects = [
      {"name": "Project 1", "url_name": "project1:index"},
      {"name": "Project 2", "url_name": "project2:index"},
      {"name": "Project 3", "url_name": "project3:index"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return render(request, 'home/index.html', context)