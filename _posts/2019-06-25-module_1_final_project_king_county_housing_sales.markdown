---
layout: post
title:      "**Module 1 – Final Project: King County Housing Sales**"
date:       2019-06-26 00:27:53 +0000
permalink:  module_1_final_project_king_county_housing_sales
---



I am currently enrolled in the Flatiron Data Science program. This is my first project, about 2 months in to the course. In this blog I will detail how I went about my exploratory data analysis, and selected the features to use in my model.

The github link to my project is: [(https://github.com/AndrewJ1/FI_Module_1/blob/master/FinalProject/dsc-1-final-project-online-ds-pt-041519/Final%20Draft.ipynb)]

The brief requirements of this project were as follows: clean, explore, and model the King County House Sales dataset with a multivariate linear regression to predict the sale price of house as accurately as possible.

In approaching this problem we used the OSEMiN Data Science model. That is – Obtain, Scrub, Explore, Model, and Interpret the data.
 


## **Obtain**
This stage is loading the required libraries, and importing the data

I also looked over the data dictionary, and size of the dataset. There are 21,597 rows and 21 columns. Starting from a very basic knowledge of real estate – larger houses cost more, location is important; there are several features that stand out.

For house size we have square footage of living space, lot, for the house and the nearest 15 neighbors. We also have number of bedrooms and bathrooms.

For location we have zipcode, latitude and longitude. We also have house that have a view to a waterfont.

## **Scrub**
In this stage I am looking for issues with the data. df.info() and df.describe() are helpful to get a quick view of the data. I notice that waterfront, view, and yr_renovated have some null values. I also see that id has some duplicates, and sqft_basement has a “?” in some rows which prevents this column from being classified as numeric. There is one house that inexplicably has 33 bedrooms and … yikes, only 1.75 bathrooms!

Here’s how I approached cleaning these data columns

**duplicate ids** – I thought about dropping the duplicates here, which appeared to be the same house selling twice (or 3x in one case) over the time period of the dataset. However I decided against removal, because the duplicates represent actual sales. If we remove the earlier sale, it may skew the total dataset in favor of more recent sales. 

**sqft_basement** – fortunately we can use other columns to populate the null values:  		
df['sqft_living'] - df['sqft_above']

**waterfront** – there are only 146 waterfront rows with the value of 1, and 2376 with null. Check the 5 point statistics of 0,1, and null for waterfront. Null and 0 have similar mean and standard deviations for price, so it is ok to replace null with zero.
df.waterfront = df.waterfront.fillna(0)

I also removed zipcode 98070 (Vashon Is) from the waterfront feature. This zipcode is different in that it is an island in Puget Sound that is only accessible by ferry. 8 of the bottom 10 rows with the waterfront feature are in zipcode 98070. 
There’s a great article about Vashon Island here. [https://www.seattletimes.com/pacific-nw-magazine/vashon-tries-to-keep-it-real-despite-an-influx-of-people-and-money/]


**view** is mainly 0, and has only 63 null values. It is ok to change the null values to 0.
df.view = df.view.fillna(0)

**yr_renovated** is also mainly 0, and requires a year to be entered for the renovation. Without any further information I change these values to 0 too.
df.yr_renovated = df.yr_renovated.fillna(0)

And I’ll drop the mystery house with 33 bedrooms.
df.drop(df[df["bedrooms"] >= 33].index, inplace = True)


## **Explore**
Histograms and Scatter plots are a good way to quickly identify distribution and relationships in the data.
 
** Location, Location, Location**
It's almost universally known that the most important 3 factors in real estate are location, location, location. Look again at the location features in our data frame: zipcode, latitude and longitude, and sqft_living15 and sqft_lot15. 

Unfortunately housing price as it relates to zipcode, latitude and longitude is not linear. You can't expect that as a zipcode increases, the house price will increase too. High priced neighborhoods can be right next to inexpensive ones, not in a line from lowest to highest. The sqft--15 features could be useful in identifying comp houses. At this moment in my training transforming these features into comps is beyond my knowledge. After running some models, I drop the sqft--15 features.

Instead I built a new feature using latitude and longitude. This uses the equirectangular distance approximation for the shortest distance between two points given their longitudes and latitudes. It's actually a fairly simple formula, despite the overbearing name.

(Note that I could have also used the more complicated Haversine formula which measures the distance on sphere. However for short distances the difference between these formulas is neglible)

I used downtown Seattle as a reference point for this fixture. Seattle is by far the largest city in King County, and proximity to work is one of the elements that make a location enticing. 

The co-ordinates for downtown Seattle are: 47.608013, -122.335167
```
# Seattle = 47.608013, -122.335167
# approx one degree of latitude ellipsoidal earth at 47N: 69.08 miles


def distance(Lat2, Long2):
    Lat1 = 47.608013
    Long1 = -122.335167
    x = Lat2 - Lat1
    y = (Long2 - Long1) * np.cos((Lat2 + Lat1)*(0.5 * np.pi/180))  
    return 69.08 * np.sqrt(x*x + y*y)

df['Miles_Seattle'] = df.apply(lambda x: distance(x['lat'], x['long']), axis=1)
df.head()

```
We now have a new column - 'Miles_Seattle' with the distance between each house and downtown Seattle

To include a directional component, I added 2 features looking at whether a house is North/South or East/West of Seattle

Some other experiments that I performed between data exploration and data modelling:

**zipcode** – I made this feature categorical, which ultimately created an extra 70 columns. Initially this looked very promising, with the model outputting a very high r2 value. However this model was almost certainly overfit, as there were a lot of errors; p-value and residual. I still felt that zipcode could be valuable to the model, so I built 2 features, which grouped the top 8 and bottom 8 zipcodes together.

**yr_built** – experimenting with this feature was an iterative process. I first grouped the years by decade. Houses built before about 1940 had more value, the mean price then dips over several decades, before rising again in recent decades. However binning yr_built by decades had a lot of errors. Ultimately I binned as follows, and had to drop pre-1940 due to large p-values.

```
bins = [1900, 1940, 1960, 1980, 2000, 2020]
bins_yr = pd.cut(df['yr_built'], bins)
bins_yr = bins_yr.cat.as_unordered()

```

**bedrooms and bathrooms** – I tested these as categories. However, this introduced extra issues, and didn’t improve the model. I ran the model with these as numeric. I still have some concerns about bedrooms / bathrooms, the maximum mean price tops out before the maximum number of rooms for example.

**grade** – gave some improvement to r2. However it was highly correlated with living size, and introduced residual errors

Before running the model I performed a log transformation and min/max scaling.

```
# log transformations to normalize data
logprice = np.log(df['price'])
loglive = np.log(df['sqft_living'])
loglot = np.log(df['sqft_lot'])
logmiles = np.log(df['Miles_Seattle'])
logbase = np.log(df['basement2']+1) #basement2 has many zero values, adding 1 so that log transformation works
bedrm = np.log(df['bedrooms'])
bathrm = np.log(df['bathrooms'])


# minmax scaling to scale the data between 0 and 1
df['price'] = (logprice-min(logprice))/(max(logprice)-min(logprice))
df['sqft_living'] = (loglive-min(loglive))/(max(loglive)-min(loglive))
df['sqft_lot'] = (loglot-min(loglot))/(max(loglot)-min(loglot))
df['Miles_Seattle'] = (logmiles-min(logmiles))/(max(logmiles)-min(logmiles))
df['basement2'] = (logbase-min(logbase))/(max(logbase)-min(logbase))
df['bedrooms'] = (bedrm-min(bedrm))/(max(bedrm)-min(bedrm))
df['bathrooms'] = (bathrm-min(bathrm))/(max(bathrm)-min(bathrm))

```

## **Model**
This blog is primarily about data exploration, so I won’t go into too much detail about the modelling. Please refer to the github link at the top of the page for full details on the model.

I ran 2 models: OLS from statsmodel, and linear regression from sklearn. While, it’s not necessary to run 2 models, I thought that at this stage of my training that it was good practice.  I was happy with the r2 squared result, and the MSE results from a train-test split and 10-fold cross validation.

qqplot looks normal for the most part, but has some evidence of being heavy-tailed, that is extreme positive and negative residuals. The model also appears to meet the homoscedasticity test, that is, residual variance remains constant as predicted values increase. However, the chart does show a tail skewed to the right. Both of these may warrant a further look at outliers. 

## **Interpretation**
Finally, what does this model tell us? The square footage of house is the most important factor in determining price. The distance from Seattle is the second most important factor, and the third most important factor is a waterfront view (which is interesting in that it is only an active feature in less than 1% of the houses in the dataset).

Another finding that I found interesting, was that more bedrooms mean less value. On the face of it this seems like nonsense. Anyone knows that a house with more bedrooms is worth more. However, we have already taken into account the square foot of the house. The model is measuring bedrooms in the presence of house size, miles from Seattle … So the model is suggesting that if you trade another room in the house for a bedroom (large family room, kitchen) you actually decrease the house’s value.

## **Advice**
I’ll end with some advice, for anyone following in my footsteps. It feels a little odd to do this, given that I am only 2 months into my study, but hopefully this helps someone.

•	Trust yourself. You have accumulated some background knowledge to the subject simply from your own life experience. Use this. Even if your assumptions prove incorrect, this is a good starting point.

•	Be prepared to change direction and drop things that previously looked promising. New analysis can lead you to a completely different conclusion than you anticipated.

•	Don’t get stuck searching for perfect. Set a target or a time limit and stick to it. It’s easy to get bogged down continually testing and refining your model, but after a certain point it’s no longer worth it.

•	Stack Overflow, Medium, Google are all a huge help. Even better, reach out to the community if you get stuck.

•	Make extensive notes as you work, it’s very time consuming to add these in later. 

