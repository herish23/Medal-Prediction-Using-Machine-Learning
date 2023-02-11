# DATASET used https://drive.google.com/uc?export=download&id=1L3YAlts8tijccIndVPB-mOsRpEpVawk7
# PREDICTION USING MACHINE LEARNING
# 1 Selection of features using Correlation Feature Selection
# 2 Predicting the medal counts using Logistic Regression Model


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("teams.csv")
nan_count = data.isna().sum()
teams = data.dropna()
print(teams)

# Partition data into 70:30
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['medals'], axis=1),
    data['medals'],
    test_size=0.3,
    random_state=0)

corrmat = X_train.corr(method='pearson')
cmap = sns.diverging_palette(220, 20, as_cmap=True)
fig, ax = plt.subplots()
fig.set_size_inches(14, 14)
sns.heatmap(corrmat, cmap=cmap)
plt.show()


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if abs(corr_matrix.iloc[i, j]) > threshold:
                print(abs(corr_matrix.iloc[i, j]), corr_matrix.columns[i], corr_matrix.columns[j])

                colname = corr_matrix.columns[j]

                col_corr.add(colname)

    return col_corr


# setting up the threshold to 80% for improved accuracy
corr_features = correlation(X_train, 0.8)
print(corr_features)
# {'events', 'athletes', 'prev_medals'}


# we do the prediction based on the features which were selected from CFE

teams1 = teams[["events", "athletes", "prev_medals"]]
print(teams1)
print(teams.corr()["medals"])

# Feature 1
# plotting prev_medals against medals
sns.lmplot(x="prev_medals", y="medals", data=teams, fit_reg=True, ci=None)
plt.show()

# Feature 2
# plotting events against medals
sns.lmplot(x="events", y="medals", data=teams, fit_reg=True, ci=None)
plt.show()

# Feature 3
# plotting athletes against medals
sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)
plt.show()

# plotting a histogram for medals pattern
teams.plot.hist(y="medals")
plt.show()

# split the data
train = teams[teams["year"] < 2010].copy()  # for train we use data before 2010
test = teams[teams["year"] >= 2010].copy()  # for test we use data after 2010

reg = LinearRegression()
# we  use the features above 0.8 threshold
predictors = ["events", "prev_medals", "athletes"]

reg.fit(train[predictors], train["medals"])
predictions = reg.predict(test[predictors])

test["predictions"] = predictions

test.loc[test["predictions"] < 0, "predictions"] = 0

test["predictions"] = test["predictions"].round()

# calculating the error between prediction an actual that occurs
error = mean_absolute_error(test["medals"], test["predictions"])
print(error)

test["predictions"] = predictions

name = input("Enter country code:")
print(test[test["team"] == name])
