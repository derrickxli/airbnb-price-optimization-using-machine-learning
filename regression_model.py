import pandas as pd
import numpy as np
from math import sqrt

#Import stuff from Sci Kit learn.
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from scitime import Estimator
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def find_optimal_prices(model, listingInfo, columnNames, minPrice, maxPrice):

    optimal_prices = {}
    pricesToTry = [float(i) for i in range(minPrice,maxPrice,10)]

    dates = ['date_2016-10-0' + str(i) for i in range(4,5)]

    listingFrame = pd.DataFrame(columns=columnNames)
    for (k,v) in listingInfo.items():
        listingFrame[k] = [v]
    listingFrame = listingFrame.fillna(0)
    listingFrame = pd.DataFrame(listingFrame.values.repeat(len(pricesToTry),axis=0), columns=listingFrame.columns)
    listingFrame['price'] = pricesToTry

    for date in dates:
        dateFrame = listingFrame
        dateFrame[date] = [1] * len(pricesToTry)
        dateFrame = np.array(dateFrame)
        print(dateFrame)
        probs = model.predict_proba(dateFrame)
        print(probs)
        predictions = list(model.predict(dateFrame))
        for i in range(len(predictions)):
            print('Prob: ', probs[i])
            print('Prediction: ', predictions[i])
            print('--------')
        highestPriceIndex = 0
        if 1 in predictions:
            highestPriceIndex = len(predictions) - 1 - predictions[::-1].index(1)
        highestPrice = pricesToTry[highestPriceIndex]
        optimal_prices[date] = highestPrice

        # listingInfo.pop(date)

    print(optimal_prices)

def evaluate_model(model, predict_set, evaluate_set):
    predictions = model.predict(predict_set)
    probs = model.predict_proba(predict_set)
    correct = 0
    for i in range(len(predictions)):
        print('Prob: ', probs[i])
        print('Prediction: ', predictions[i])
        print('Actual: ', evaluate_set[i])
        print('--------')
        if predictions[i] == evaluate_set[i]:
            correct += 1
    print(correct / float(len(predictions)))

def main():
    #Import listing data.
    pricesToTry = [float(i) for i in range(50,150,5)]
    print(pricesToTry)
    listingData = pd.read_csv('./data/listings.csv')

    #Select features we want to use.
    features = ['id','host_since','host_response_time','host_response_rate','host_acceptance_rate','host_is_superhost','host_listings_count','host_total_listings_count','host_verifications','host_has_profile_pic','host_identity_verified','neighbourhood_group_cleansed','city','zipcode','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']

    simple_features = ['id','neighbourhood_group_cleansed','zipcode','property_type','room_type','accommodates','bathrooms','bedrooms','beds','price']

    #Keep the features we want to use.
    listingData = listingData[simple_features]

    #Add in the calendar data to the listing data.
    listingData = pd.DataFrame(listingData.values.repeat(365,axis=0), columns=listingData.columns)

    calendarData = pd.read_csv('./data/calendar.csv')
    calendarData.rename(columns={'price':'daily_price'}, inplace=True)

    listings = pd.concat([listingData, calendarData], axis=1)

    listings['final_price'] = listings['daily_price'].combine_first(listings['price'])
    listings.drop(columns=['price', 'daily_price','id','listing_id'],inplace=True)

    #Transform categorical features to boolean features using one hot encoding.
    listings = pd.get_dummies(listings, columns=['date', 'neighbourhood_group_cleansed', 'zipcode','property_type','room_type'])

    #Format price variable to remove dollar sign and commas.
    listings.rename(columns={'final_price':'price'}, inplace=True)
    listings['price'] = listings['price'].str.replace('\$|,', '')
    listings['price'] = pd.to_numeric(listings['price'])

    #Split and format the data into features, labels, training set, and test set.
    # listings = listings.truncate(after=36500)

    listings = listings.applymap(lambda x: 1 if x == 't' else x)
    listings = listings.applymap(lambda x: 0 if x == 'f' else x)
    listings['price'] = [-val for val in list(listings['price'])]

    y = listings['available']
    x = listings.drop(columns=['available'])

    trainX, testX, trainY, testY = train_test_split(x,y,random_state=0)

    trainX = np.array(trainX)
    testX = np.array(testX)
    trainY = np.array(trainY)
    testY = np.array(testY)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 15, num = 5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 15, num = 5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


    # rf = make_pipeline(Imputer(), StandardScaler(), RandomForestClassifier(n_estimators=25, random_state=42, verbose=1))

    rf2 = RandomForestClassifier(random_state=42, verbose=1)

    rf_random = GridSearchCV(rf2, random_grid, cv = 5, verbose=2)

    # kn = make_pipeline(Imputer(), StandardScaler(), KNeighborsClassifier(20,weights='uniform'))
    # log = make_pipeline(Imputer(), StandardScaler(), LogisticRegression())
    # las = make_pipeline(Imputer(), StandardScaler(), Lasso(alpha=0.1))
    # rid = make_pipeline(Imputer(), StandardScaler(), Ridge(alpha=1.0))
    # svm = make_pipeline(Imputer(), StandardScaler(), SVC(gamma='scale'))

    # las.fit(trainX,trainY)
    # rid.fit(trainX,trainY)
    # kn.fit(trainX,trainY)
    # SGDc.fit(trainX,trainY)
    # rf.fit(trainX,trainY)
    rf_random.fit(trainX,trainY)
    # svm.fit(trainX,trainY)
    # log.fit(trainX,trainY)

    # importances = rf.steps[2][1].feature_importances_
    # feature_importances = pd.DataFrame({"feature":x.columns.values, "importance":importances})
    # feature_importances = feature_importances.sort_values("importance", ascending=False).head(22)
    # print(feature_importances)

    # evaluate_model(rid, trainX, trainY)
    # evaluate_model(rid, testX, testY)

    # evaluate_model(las, trainX, trainY)
    # evaluate_model(las, testX, testY)
    # evaluate_model(kn, trainX, trainY)
    # evaluate_model(kn, testX, testY)
    # evaluate_model(SGDc, trainX, trainY)
    # evaluate_model(SGDc, testX, testY)
    # evaluate_model(rf, trainX, trainY)
    # evaluate_model(rf, testX, testY)
    evaluate_model(rf_random.best_estimator_, trainX, trainY)
    evaluate_model(rf_random.best_estimator_, testX, testY)
    # evaluate_model(svm, trainX, trainY)
    # evaluate_model(svm, testX, testY)

    # evaluate_model(log, trainX, trainY)
    # evaluate_model(log, testX, testY)

    # listingInfo = {'neighbourhood_group_cleansed_Queen Anne': 1,'zipcode_98119': 1,'property_type_House': 1,'room_type_Entire home/apt': 1,'accommodates': 8,'bathrooms': 3,'bedrooms': 4,'beds': 4}
    # find_optimal_prices(log, listingInfo, list(x.columns), 0, 1000)

if __name__ == '__main__':
	main()
