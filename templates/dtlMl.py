import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

df = pd.read_csv("E:/dtl-project/Dataset/Fertilizer_Prediction.csv")


y = df['Fertilizer Name'].copy()
X = df.drop('Fertilizer Name', axis=1).copy()

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3, 4])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, shuffle=True, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(
    n_estimators=100, criterion='gini', random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

encode_soil = LabelEncoder()
df['Soil Type'] = encode_soil.fit_transform(df['Soil Type'])

# creating the DataFrame
Soil_Type = pd.DataFrame(zip(encode_soil.classes_, encode_soil.transform(encode_soil.classes_)),
                         columns=['Original', 'Encoded'])
Soil_Type = Soil_Type.set_index('Original')

encode_crop = LabelEncoder()
df['Crop Type'] = encode_crop.fit_transform(df['Crop Type'])

# creating the DataFrame
Crop_Type = pd.DataFrame(zip(encode_crop.classes_, encode_crop.transform(encode_crop.classes_)),
                         columns=['Original', 'Encoded'])
Crop_Type = Crop_Type.set_index('Original')

encode_ferti = LabelEncoder()
df['Fertilizer Name'] = encode_ferti.fit_transform(df['Fertilizer Name'])

# creating the DataFrame
Fertilizer = pd.DataFrame(zip(encode_ferti.classes_, encode_ferti.transform(encode_ferti.classes_)),
                          columns=['Original', 'Encoded'])
Fertilizer = Fertilizer.set_index('Original')

x_train, x_test, y_train, y_test = train_test_split(df.drop('Fertilizer Name', axis=1), df['Fertilizer Name'],
                                                    test_size=0.2, random_state=1)

rand = RandomForestClassifier(random_state=42)
rand.fit(x_train, y_train)

pred_rand = rand.predict(x_test)

params = {
    'n_estimators': [300, 400, 500],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 8]
}
grid_rand = GridSearchCV(rand, params, cv=3, verbose=3, n_jobs=-1)
grid_rand.fit(x_train, y_train)
pred_rand = grid_rand.predict(x_test)

pickle_out = open('classifier.pkl', 'wb')
pickle.dump(grid_rand, pickle_out)
pickle_out.close()

model = pickle.load(open('classifier.pkl', 'rb'))
ans = model.predict([[34, 65, 62, 0, 1, 7, 9, 30]])
if ans[0] == 0:
    print("10-26-26")
elif ans[0] == 1:
    print("14-35-14")
elif ans[0] == 2:
    print("17-17-17	")
elif ans[0] == 3:
    print("20-20")
elif ans[0] == 4:
    print("28-28")
elif ans[0] == 5:
    print("DAP")
else:
    print("Urea")
