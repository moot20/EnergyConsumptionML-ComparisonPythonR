library(caret)
library(cluster)

data <- read.csv('datasets/online_shoppers_intention.csv')

data$Month <- as.numeric(factor(data$Month))
data$VisitorType <- as.numeric(factor(data$VisitorType))
data$Weekend <- as.numeric(data$Weekend)
data$Revenue <- as.numeric(data$Revenue)

features <- data[, !names(data) %in% c("Revenue")]

preproc <- preProcess(features, method=c("center", "scale"))
scaled_features <- predict(preproc, features)

n_clusters <- 3
set.seed(42)
kmeans_model <- kmeans(
    scaled_features, 
    centers = n_clusters,
    algorithm = "Lloyd",
    iter.max = 300,
    nstart = 10
)