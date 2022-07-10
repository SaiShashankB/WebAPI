import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
df = pd.read_excel("Social_Network_Ads1.xlsx")
print(df.head())
df.drop("User ID", axis=1, inplace=True)
gender_dummies = pd.get_dummies(df.Gender, prefix="Gender")
df_with_dummies = pd.concat([df, gender_dummies], axis="columns")
df_with_dummies.drop("Gender", axis="columns", inplace=True)
df_with_dummies.drop("Gender_Female", axis=1, inplace=True)
X = df_with_dummies.drop("Purchased", axis=1)
print(X.head())
y = df_with_dummies["Purchased"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
pickle.dump(classifier, open("model.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))