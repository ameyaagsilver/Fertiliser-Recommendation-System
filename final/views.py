from requests import api
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import geoip2.database


from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponse 
from django.conf import settings
import ipinfo
from pprint import pprint
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  
import warnings
import requests

warnings.filterwarnings("ignore")
from django.shortcuts import render


def get_ip_details(ip_address=None):
	ipinfo_token = getattr(settings, "IPINFO_TOKEN", None)
	ipinfo_settings = getattr(settings, "IPINFO_SETTINGS", {})
	ip_data = ipinfo.getHandler(ipinfo_token, **ipinfo_settings)
	ip_data = ip_data.getDetails(ip_address)
	return ip_data


def init_data(request):
    print("*******************")
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    print(ip)
    print("*********************")

    print("$$$$$$$$")
    ip_data = requests.get("http://ipinfo.io/"+ip+"?token=c2a5a39278141a")
    city = ip_data.json()["city"]
    if "Chandīgarh" == city:
        city = "Chandigarh"
    pprint(ip_data.json())
    print("##########")

    #location to weather data through openweatherAPI
    userapi = "a51a2c05927340d656097f79ec64360c"
    completeAPILink = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+userapi
    print(completeAPILink)
    api_link = requests.get(completeAPILink)
    api_data = api_link.json()
    if api_data['cod'] == '404':
        print("Error")
    else:
        pprint(api_data)
    
    return [ip_data, api_data]

def home(request):
    return render(request, 'index.html')

def recommend(request):
    ip_data, api_data = init_data(request)
    temperature = int(api_data['main']['temp'] - 273.15)
    humidity = api_data['main']['humidity']
    return render(request, 'recommend.html', {'temperature': temperature, 'humidity':humidity})

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def faq(request):
    return render(request, 'faq.html')

def services(request):
    return render(request, 'services.html')

def output(request):
    
    ip_data, api_data = init_data(request)
    temperature = int(api_data['main']['temp'] - 273.15)
    humidity = api_data['main']['humidity']
    # inp1=request.POST.get('p1')
    # inp2=request.POST.get('p2')
    inp3=request.POST.get('p3')
    inp4=request.POST.get('p4')
    inp5=request.POST.get('p5')
    inp6=request.POST.get('p6')
    inp7=request.POST.get('p7')
    inp8 = request.POST.get('p8')
    # inp1=int(inp1)
    inp1=temperature
    # inp2 = int(inp2)
    inp2=humidity
    inp3=int(inp3)
    inp4=int(inp4)
    inp5=int(inp5)
    inp6=int(inp6)
    inp7=int(inp7)
    inp8=int(inp8)
    model = pickle.load(open('classifier.pkl', 'rb'))
    ans = model.predict([[inp1, inp2, inp3,inp4 , inp5, inp6, inp7, inp8]])
    print([inp1, inp2, inp3, inp4, inp5, inp6, inp7, inp8])
    ph1=""
    p1=""
    weather_data = "Temperature = "+str(inp1)+" Humidity = "+str(inp2)
    if ans[0] == 0:
        data="10-26-26"
        desc1 = "10:26:26 is a complex fertiliser containing all the three major plant nutrients viz. Nitrogen, Phosphorous and Potassium."
        desc2 = "10:26:26 contains Phosphorous and Potassium in the ratio of 1:1, the highest among the NPK fertilisers. It contains 7% Nitrogen in the Ammonical form, 22% out of 26% phosphate in the water soluble form and the entire 26% potash is available in the water soluble form."
        desc3 = "10:26:26 is ideally suitable for crops which require high phosphate and potassium and this grade is very popular among the Sugarcane farmers of Maharashtra, Karnataka and Andhra Pradesh and Potato farmers of West Bengal & Uttar Pradesh."
        desc4 = ":26:26 is also suitable for Fruit crops."
    elif ans[0] == 1:
        data="14-35-14"
        desc1 ="Highest total nutrient content among NPK fertilizers (63 %)."
        desc2 = "N & P ratio same as DAP. In addition, GROMOR 14-35-14 has extra 14% potash."
        desc3 = "High in Phosphorous content (35%)."
        desc4 = "Best for almost all kinds of crops like Cotton, Groundnut, Chilly, Soya bean, Potato."
    elif ans[0] == 2:
        data="17-17-17	"
        desc1 = "Gromor 17:17:17 is a complex fertiliser containing all three major plant nutrients viz. Nitrogen, Phosphorous and Potassium in equal proportion."
        desc2 = "Supplies all three major nutrients 17% each of nitrogen, phosphate & potash to the crops."
        desc3 = "Supplies 17% each of nitrogen, phosphate & potash."
        desc4 = "Contains 14.5% out of 17% phosphate in water soluble form which is easily available to crops."
    elif ans[0] == 3:
        data="20-20-0"
        desc1 = "This is the most popular grade among the farming community."
        desc2 = "It contains 20% Nitrogen. Of this 90% of Nitrogen is present in Ammonical form and the rest in Amide form. However, the entire Nitrogen is available to crops in Ammonical form."
        desc3 = "It is granular in nature and can be easily applied by broadcasting, placement or drilling."
        desc4 = "It is an excellent fertiliser for all crops grown in Sulphur deficient soils and is highly suitable for Sulphur loving crops such as Oil seeds."
    elif ans[0] == 4:
        data="28-28-0"
        desc1 = "Complex with highest N & P in 1:1 ratio."
        desc2 = "Unique granulation by coating prilled urea with Ammonium Phosphate layer."
        desc3 = "It does not contain any filler and it has 100% nutrient containing material having secondary and micronutrients such as Sulphur, Calcium and Iron."
        desc4 = "It is an ideal complex fertiliser for all crops for basal application."
    elif ans[0] == 5:
        data="DAP"
        ph1 = "Diammonium Phosphate";
        p1 = "It is the most popular phosphatic fertiliser because of its high analysis and good physical properties.The composition of DAP is N-18% and P2O5 -46%."
        desc1 = "As far as Indian farmer is concerned, IFFCO's DAP is not just a source of crucial nutrients N, P for the crops, but is an integral part of his/her quest for nurturing mother earth."
        desc2="The bountiful crop that results from this care is an enough reason for the graceful bags of IFFCO DAP bags to be an integral part of the farmers's family."
        desc3=" The Indian farmer's confidence and trust stems from the fact that IFFCO's DAP is a part of a complete package of services, ably supported by a dedicated team of qualified personnel."
        desc4="This fertiliser is useful for all kinds of crops."
    else:      
        data="Urea"
        desc1 = "Urea is the most important nitrogenous fertiliser in the country because of its high N content (46%N)."
        desc2 = "Besides its use in the crops, it is used as a cattle feed supplement to replace a part of protein requirements."
        desc3 = "It has also numerous industrial uses notably for production of plastics. Presently all the Urea manufactured in the country is Neen coated."
        desc4 = "Urea is a raw material for the manufacture of two main classes of materials: urea-formaldehyde resins and urea-melamine-formaldehyde used in marine plywood."

    return render(request, 'recommend.html', {'weather':weather_data, 'ipdata':ip_data, 'data': data, 'desc1': desc1, 'desc2': desc2, 'desc3': desc3, 'desc4': desc4, 'ph1': ph1, 'p1': p1})
