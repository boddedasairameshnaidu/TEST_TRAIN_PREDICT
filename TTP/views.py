from django.shortcuts import render
from django.http import HttpResponseBadRequest
from django.http import HttpResponse, Http404
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from django.core.files.storage import FileSystemStorage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from .models import *
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import os
# Create your views here.

def is_text_column(series):
    return pd.api.types.is_string_dtype(series)

def to_dense(X):    
    # Converts a sparse matrix to a dense matrix.
    return X.toarray() if hasattr(X, 'toarray') else X

def home(request):
    return render(request, 'home.html')

def train(request):
    return render(request, 'train.html')

def traindata(request):
    if request.method == 'POST':
        if 'doc' in request.FILES:
            uploaded_file = request.FILES['doc']
            file_name = uploaded_file.name
            request.session["file"] = file_name
            print(uploaded_file)
            #save file
            fs = FileSystemStorage()
            if fs.exists(file_name):
                fs.delete(file_name)
            fs.save(uploaded_file.name, uploaded_file)
            # uploaded_file_url = fs.url(filename)
        else:
            file_name=None
    
    if "file" in request.session:
        # write code here
        file_name = request.session["file"]
        datatype = request.POST['data_type']
        classData = request.POST['class_col']
        request.session['dataType'] = datatype
        request.session['classdata'] = classData
        df = pd.read_csv(file_name)
        df = df.dropna()
        df = df.drop_duplicates()
        y = df[classData]
        X = df.drop(classData, axis=1)
        # Convert X to a Series if it has only one column
        if X.shape[1] == 1:
            X = X.squeeze()
        names = ['nb','nn','knn','rf','dt','svc','lr']
        alg = [MultinomialNB(),MLPClassifier(),KNeighborsClassifier(),RandomForestClassifier(),DecisionTreeClassifier(),SVC(),LogisticRegression()]
        print(datatype)
        if datatype == 'numeric':
            stz = False
            for i in range(len(alg)):
                clf = alg[i]
                model = clf.fit(X,y)
                pickle.dump(model,open(names[i]+'.sav','wb'))
                stz = True
            msg = 'DataSet trained successfully..Now you can test your dataSet!'
        else:
            stz = False
            print(X, y)
            for i in range(len(alg)):
                tfidf = TfidfVectorizer(stop_words='english',use_idf=True,smooth_idf=True)
                pipeline = Pipeline([('classifier',tfidf),('alg',alg[i])])
                pickle.dump(pipeline.fit(X,y),open(names[i]+'.sav','wb'))
                stz = True
            msg = 'DataSet trained successfully..Now you can test your dataSet!'
    return render(request,'train.html',{'msg':msg,'stz':stz})  



def test(request):
    return render(request,'test.html')

plt.switch_backend('Agg')

def testdata(request):
    disp = False
    stz = False
    if request.method == 'POST':
        if 'test_doc' in request.FILES:
            uploaded_file = request.FILES['test_doc']
            file_name = uploaded_file.name
            request.session["test_file"] = file_name
            print(uploaded_file)
            #save file
            fs = FileSystemStorage()
            if fs.exists(file_name):
                fs.delete(file_name)
            fs.save(uploaded_file.name, uploaded_file)
            # uploaded_file_url = fs.url(filename)
        else:
            file_name=None
    # Write code here
    if "test_file" in request.session and 'file' in request.session:
        file_name = request.session["test_file"]
        datatype = request.session['dataType']
        # datatype = request.session.get('dataType')
        classData = request.session['classdata']
        
        if datatype == 'numeric' or datatype == 'text':
            df = pd.read_csv(file_name)
            df = df.dropna()
            df = df.drop_duplicates()
            y = df[classData]
            X = df.drop(classData,axis=1)
            # Convert X to a Series if it has only one column
            if X.shape[1] == 1:
                X = X.squeeze()
            algNames = ['nb.sav','nn.sav','knn.sav','rf.sav','dt.sav','svc.sav','lr.sav']
            names = ['nb','nn','knn','rf','dt','svc','lr']
            algo = ['NaiveBayes','NeuralNetworks','KNN','RandomForest','DecisionTree','SVM','LogisticRegression']
            d = Res.objects.all()
            d.delete()
            for i in range(len(algNames)):
                model = pickle.load(open(algNames[i],'rb'))
                y_pred = model.predict(X)
                acc = accuracy_score(y,y_pred)
                pre = precision_score(y,y_pred,average='micro')
                rec = recall_score(y,y_pred,average='micro')
                f1 = f1_score(y,y_pred,average='micro')
                d = Res(alg_name=names[i],acc=acc,pre=pre,rec=rec,f1=f1)
                d.save()
                disp = True            
        res = Res.objects.all()
        acc = []
        pre = []
        rec = []
        f1 = []
        algNames = ['nb.sav','nn.sav','knn.sav','rf.sav','dt.sav','svc.sav','lr.sav']    
        for i in res:
            acc.append(i.acc)
            pre.append(i.pre)
            rec.append(i.rec)
            f1.append(i.f1)

        scores = [acc, pre, rec, f1]
        labels = ['Accuracy','Precision','Recall','F1']
        # print(pre)
        for i in range(4):
            plt.bar(names, scores[i],color=['violet','indigo','blue','green','yellow','orange','red'])
            plt.xlabel('ALGORITHMS')
            plt.ylabel('SCORES')
            plt.title(labels[i])
            plt.savefig(f'C:\\Users\\RAMESH\\Desktop\\CLOUD DJANGO + ML\\ML\\TTP\\static\\assets\\images\\{labels[i]}.png')
        
        mx = max(acc)
        idx = acc.index(mx)
        algorithm = algo[idx]
        request.session['algPredict'] = algNames[idx]
        
        msg = 'Testing done successfully.View Performance metrics below...'
        msg1 = f'Since, {algorithm} gave best results.This will be used for predicting your Data.'
        stz = True
        print(idx)
    return render(request, 'test.html', {'disp': disp,'msg':msg, 'msg1':msg1,'stz':stz})

def predict(request):
    return render(request, 'predict.html')

def predictData(request):
    if request.method == 'POST':
        stz = False
        data = request.POST['listInput']
        datatype = request.session['dataType']
        if datatype == 'numeric':
            data = [float(x) for x in data.split()]
            # Convert the list to a 2D array as expected by the model
            data = np.array(data).reshape(1, -1)
        else:
            data = data.split(',')
        print(data)
        # session_items = {key: value for key, value in request.session.items()}
        # print("Session Data:", session_items)
        # print(request.session)
        if 'algPredict' in request.session:
            alg = request.session['algPredict']
            print(alg)
            model = pickle.load(open(alg,'rb'))
            res = model.predict(data)
            msg = f'{res}'    
            stz = True   
    return render(request,'predict.html',{'msg':msg,'stz':stz})

def result(request):
    if 'file' in request.session:
        d = Res.objects.all()
        return render(request, 'result.html',{'disp':d})
def admlogin(request):
    return render(request, 'login.html')

def register(request):
    return render(request,'register.html')

def registerstore(request):
    name = request.POST['name']
    email = request.POST['email']
    pwd = request.POST['pwd']
    d = User(name=name,email=email,pwd=pwd)
    d.save()
    msg = "Registered Successfully...Now Login"
    stz = True
    return render(request,'login.html',{'msg':msg,'stz':stz})        

def admloginaction(request):
    name = request.POST['name']
    pwd = request.POST['pwd']
    d = User.objects.filter(name=name).filter(pwd=pwd).count()
    if d>0:
        request.session["UserName"] = name
        return render(request,'admin_home.html')
    else:
        msg = "Invalid Credentials"
        stz = True
        return render(request,'login.html',{'msg':msg,'stz':stz})

def adminhome(request):
    return render(request,'admin_home.html')

def logout(request):
    if "UserName" in request.session:
        del request.session["UserName"]
    else:
        pass
    return render(request,'home.html')