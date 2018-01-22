# Basic 
````
x = 1:5
import numpy as np
np.arange(1,6,1)

matrix(list,ncol=3) # byrow=T
list.reshape(3,3) #

y = matrix[,1] # R start with 1
y = matrtix[:,0] #python start with 0

cbind, rbind
np.vstack  np.hstack  np.append(x,y, axis=1)

columnname(x) <- c("columnname1","columnname2")
rowname(x)<- c("row1","row2")

dt = np.dtype({'names':['1st','2nd','3rd'],'formats':['f8','f8','f8']})
x = np.array(obj, dt)
````
# Importing a CSV
````
nba <- read.csv("nba_2013.csv",header=True)
nba <- read.table("nba_2013.csv",header=FALSE, sep=",")

import pandas
nba = pandas.read_csv("nba_2013.csv",header=1)
````
# Finding the number of rows
````
dim(nab)

nba.shape()
````
# First row or top N row
````
head(nba,1)

nba.head(1)
````
# Split into training and testing sets
````
trainRowCount <- floor(0.8 * nrow(nba))
set.seed(1)
trainIndex <- sample(1:nrow(nba), trainRowCount)
train <- nba[trainIndex,]
test <- nba[-trainIndex,]

train = nba.sample(frac=0.8, random_state=1)
test = nba.loc[~nba.index.isin(train.index)]
or
trainin = pima[0::2,:8]   #get each other row with the frist eight columns
test = pima[1::4,:8]
vaild= pima[3::4,:8]
````
# the average of each statistic
````
sapply(nba, mean, na.rm=TRUE)  # na.rm=TRUE means ignore NA
apply(nba, 1, mean) # 1= row 2=column
# extract data, renturn a list
# 3rd param= num of row, 4th param=num of column
lapply(MyList,"[", 1, ) 
#wrapper function of lapply()
sapply(MyList,"[", 2, 1 )

nba.mean() # already ignore NA
map(function,list) or 
map(lambda x: x[1]*2 + 3, [[1,2,3], [1,4]])
np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int) #Construct an array by executing a function over each coordinate.  #lambda is anonymous function in python
````
# scatterplots
````
library(GGally)  # ggplot
ggpairs(nba[,c("ast", "fg", "trb")])

import seaborn as sns  
import matplotlib.pyplot as plt # matplotlib
sns.pairplot(nba[["ast", "fg", "trb"]])
plt.show()
````
# K-means cluster
````
library(cluster)
set.seed(1)
isGoodCol <- function(col){
   sum(is.na(col)) == 0 && is.numeric(col) # remove non-numeric columns and columns with missing values
}  # python numpy.isnan
goodCols <- sapply(nba, isGoodCol)
clusters <- kmeans(nba[,goodCols], centers=5)
labels <- clusters$cluster

from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = nba._get_numeric_data().dropna(axis=1) # remove non-numeric columns and columns with missing values
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
````
# PCA
````
nba2d <- prcomp(nba[,goodCols], center=TRUE)
twoColumns <- nba2d$x[,1:2]
clusplot(twoColumns, labels)

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()
````
# univariate linear regression
````
fit <- lm(ast ~ fg, data=train)
predictions <- predict(fit, test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[["fg"]], train["ast"])
predictions = lr.predict(test[["fg"]])
````
# summary of stats of module
````
summary(fit)

import statsmodels.formula.api as sm
model = sm.ols(formula='ast ~ fga', data=train)
fitted = model.fit()
fitted.summary()
````
# random forest 
````
library(randomForest)
predictorColumns <- c("age", "mp", "fg", "trb", "stl", "blk")
rf <- randomForest(train[predictorColumns], train$ast, ntree=100)
predictions <- predict(rf, test[predictorColumns])

from sklearn.ensemble import RandomForestRegressor
predictor_columns = ["age", "mp", "fg", "trb", "stl", "blk"]
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(train[predictor_columns], train["ast"])
predictions = rf.predict(test[predictor_columns])
````
# error
````
mean((test["ast"] - predictions)^2) # MSE

from sklearn.metrics import mean_squared_error
mean_squared_error(test["ast"], predictions)
````
# Download a webpage
````
library(RCurl)
url <- "http://www.basketball-reference.com/boxscores/201506140GSW.html"
data <- readLines(url)

import requests
url = "http://www.basketball-reference.com/boxscores/201506140GSW.html"
data = requests.get(url).content
````
# Extract player box scores
````
library(rvest)
page <- read_html(url)
table <- html_nodes(page, ".stats_table")[3]
rows <- html_nodes(table, "tr")
cells <- html_nodes(rows, "td a")
teams <- html_text(cells)

extractRow <- function(rows, i){
    if(i == 1){
        return
    }
    row <- rows[i]
    tag <- "td"
    if(i == 2){
        tag <- "th"
    }
    items <- html_nodes(row, tag)
    html_text(items)
}

scrapeData <- function(team){
    teamData <- html_nodes(page, paste("#",team,"_basic", sep=""))
    rows <- html_nodes(teamData, "tr")
    lapply(seq_along(rows), extractRow, rows=rows) 
}

data <- lapply(teams, scrapeData)

from bs4 import BeautifulSoup
import re
soup = BeautifulSoup(data, 'html.parser')
box_scores = []
for tag in soup.find_all(id=re.compile("[A-Z]{3,}_basic")):
    rows = []
    for i, row in enumerate(tag.find_all("tr")):
        if i == 0:
            continue
        elif i == 1:
            tag = "th"
        else:
            tag = "td"
        row_data = [item.get_text() for item in row.find_all(tag)]
        rows.append(row_data)
    box_scores.append(rows)
````
# plot configuration
````
par(mfrow=c(2,2))   # Set graphics parameter
dev.new()  ## opens a new window
plot(lm.out,ask=F)  ## Plots the fitted curve


````