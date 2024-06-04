from django.urls import path
from . import views

urlpatterns = [
    path('',views.home,name="home"),
    path('train/',views.train,name="train"),
    path('traindata/',views.traindata,name="traindata"),
    # path('download_model/<str:model_name>/', views.download_model, name='download_model'),
    path('admlogin/',views.admlogin,name="admlogin"),
    path('register/',views.register,name="register"),
    path('registerstore/',views.registerstore,name="registerstore"),
    path('admloginaction/',views.admloginaction,name="admloginaction"),
    path('adminhome/',views.adminhome,name="adminhome"),
    path('logout/',views.logout,name="logout"), 
    path('test/',views.test,name="test"),
    path('testdata/',views.testdata,name="testdata"),
    path('predict/',views.predict,name="predict"),    
    path('predictData/',views.predictData,name="predictData"),
    path('result/',views.result,name='result'),
]

