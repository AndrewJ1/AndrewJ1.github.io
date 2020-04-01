---
layout: post
title:      "Machine Learning with NFL Data"
date:       2020-04-01 23:30:18 +0000
permalink:  machine_learning_with_nfl_data
---


For my recent project on Machine Learning I was able to choose my own dataset. I was immediately drawn towards investigating NFL data, as this is my favorite American sport, and until recently there hasn’t been much opportunity to perform detailed analysis at the lowest level (not without paying for data, anyway).

I was delighted to find a recent Kaggle competition with a lavish abundance of data on rushing plays. Laid out in a table were almost 700,000 glorious rows. Packed full with player information – weight, height, age, speed, x and y co-ordinates … even the college that each player attended. Team, stadium, weather was there too. Better yet, this is current data, right up to the latest season. I can remember watching many of these games! 

There was only one problem – the data was a regression problem, and I needed it to be classification.


#### About the data

The dataset I used is available on Kaggle. It was the basis for the NFL Big Data Bowl competition – the goal of which was to predict to predict how many yards a team will gain on each rushing play. The dataset contains game, play, and player-level data, including the position and speed of players as provided in the NFL’s Next Gen Stats data. These features known at the time when the ball is handed off. Each row in the file corresponds to a single player's involvement in a single play.

[Big_Data_Bowl](http://www.kaggle.com/c/nfl-big-data-bowl-2020/data)


#### How to turn a regression problem into a classification one

So, how could I transform this dataset into a classifier problem? I have some domain knowledge from watching football games (certainly not as much as your average hard-core fantasy football devotee). As I did some research, a classic football analytics book came to the surface. The Hidden Game of Football was published in 1988. It was one of the first statistical approach to analyzing American football.

[Hidden_Game_of_Football](http://www.amazon.com/Hidden-Game-Football-Bob-Carroll/dp/0446514144)


The book discusses how to measure the success of each play with a simple formula:
* 	    1st Down: need to gain 40% of the yards needed for a first down to be successful
* 	    2nd Down: gain 60% of the remaining yards needed for a first down
* 	    3rd / 4th Down: gain a first down

How well would a 32-year-old formula hold up to the modern explosive game? Well, the leading rusher in 1988 was Eric Dickerson with 1659 yards. The leading rusher in 2019 was Derrick Henry with 1540 yards … so there’s a chance.

```
def success_conditions(s):
    if (s['Down'] == 1) & (s['Yards'] >= s['Distance'] * 0.4):
        return 1
    elif (s['Down'] == 2) & (s['Yards'] >= s['Distance'] * 0.6):
        return 1
    elif s['Yards'] >= s['Distance']:
          return 1
    else:
        return 0
				
				
df_rush['Success'] = df_rush.apply(success_conditions, axis=1)
```


This give us 14000 successes out 31000 plays, or 45%. It is a nicely balanced dataset, so we don’t need to use any imbalanced data techniques. This metric is also remarkably balanced by Season, Quarter, and Down. Only 4th Down is more than +/- 3%, because teams won’t even attempt to run on a 4th Down unless they know that have a very good chance of success. So, the metric holds up, and we’ll … er … run with it.

![Success by Down](https://imgur.com/KFHY9lu)

So that I can experiment more easily with the Machine Learning part of this project I’m going to simplify the dataset. I include all the plays, but focus on the rusher only – and not the remaining 21 players on the field. I include the offensive and defensive formations, but not the individual players. This reduces my dataset from almost 700,000 rows to 31,000.

#### Now for the Machine Learning

First step is to prepare the data. Among other things this means: One hot encode the categorical features; standardize the data (subtract the mean, and divide by the standard deviation); split the data into a training and a test set. Fortunate scikit-learn has tools to help you handle these steps easily. Here's one example:

```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32, test_size=0.2)
```

We’re now ready to run a base model. 
```
lr_clf = LogisticRegression(random_state=32, max_iter=5000).fit(X_train, y_train)
```


Logistic Regression gets me 63% right of the bat. Now that we have a starting point, let’s try to improve it.
In this step I build various models using different Machine Learning algorithms and compare between them. All of them seem have to exotic sounding names: Decision Trees, Support Vector Machines, Random Forest, AdaBoost, XGBoost.
Once again scikit-learn comes to the rescue with tools to help me cross validate the date; that split the data into subsets called folds and train and evaluate each. Tuning the hyperparameters using grid search, and putting this altogether in a pipeline.

I compare the performance of the models using F1 score. This gives the model a balance between precision and recall. Precision is the percentage of positive results which are actually positive, recall refers to the percentage of actual positive results that are correctly classified by the model. There is a trade-off between these measures. Using F1 instead of accuracy avoids the problem of focusing on one method at the expense of the other.

Note a further methodology that I have chosen not to include in dimensionality reduction. I chose not to do this because I want to interpret the features after running the models

It is the boosting methods that get the best result – XGBoost is the best of all. After a couple of extra rounds of tuning I get a final result.  Word of warning: It’s easier to drive yourself crazy here, making endless tweaks for increasingly smaller and smaller benefit.

```
xgboost_clf = XGBClassifier(colsample_bytree= 0.4,
                            gamma= 0.01, 
                            learning_rate= 0.11, 
                            max_depth= 5, 
                            min_child_weight= 2,
                            n_estimators= 93,
                            subsample= 0.8,
                            random_state = 32)

xgboost_clf.fit(X_train, y_train)
training_preds = xgboost_clf.predict(X_train)
xgboost_preds = xgboost_clf.predict(X_test)

print(f"F1 score Score: {(f1_score(y_test, xgboost_preds)) :.2%}")
print(f"Training Accuracy: {(accuracy_score(y_train, training_preds)) :.2%}")
print(f"Test Accuracy: {(accuracy_score(y_test, xgboost_preds)) :.2%}")
print(f"Precision Score: {(precision_score(y_test, xgboost_preds)) :.2%}")
print(f"Recall Score: {(recall_score(y_test, xgboost_preds)) :.2%}")
print('------------------')

# print the classification report with all scoring results 
print(classification_report(y_test, xgboost_preds))

# print confusion matrix (basic)
print(confusion_matrix(y_test,xgboost_preds))
```

The model results in an accuracy of 66% and F1 score of 60%. The model was better at precision, than recall, as it did not predict success on a number of successful plays (False Negatives). So, it would be more useful for an offensive coordinator, than their defensive counterpart -  in that it provides a reasonable guarantee of success, but it is not as useful in warning of plays to be aware of.

Obviously, my model is not accurate enough to win a Superbowl, or even a fantasy league for that matter. But it was a great way to learn about machine learning, and it provided some unexpected insights.




