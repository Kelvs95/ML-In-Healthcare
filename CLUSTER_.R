# Importing Libraries

#install.packages("NbClust")
library(tidyverse)
#install.packages("factoextra")
library(factoextra)    
#install.packages("NbClust")
library(NbClust)
#install.packages("cluster")
library(cluster)

options(repr.plot.width=20, repr.plot.height=12) #setting the figure and chart sizes


##### Import Dataset ###########

heart <- read_csv("heart-c.csv")
heart <- heart %>% drop_na() #removing NA/incomplete data in the dataset 
heart$...1 <- NULL
summary(heart)

############ QUESTION 1################
# 1.	Why should the attribute “class” in heart-c.csv (“num”) not be included for clustering?

# Clustering is an unsupervised classfication, the class ("num") creates a pre-determined disease
# classes of "no disease" for class <50 and "increased level of heart disease" for class >50_1
# thereby introducing a bias to the unsupervised  classification.

heart_data <- heart
heart_data$num <- NULL #removing the attribute class in the column num

######## 3.Which features would you expect to be less useful when using K-means and why? ####

# Clustering is generally applied to continuous data or numerical data, 
# because it seeks to detect commonalities and group data points based on numerical features.
# our dataset has a mix of both categorical and continuous. The continuous variables in our 
# dataset includes age, trestbps, chol, thalach, oldpeak, and ca. These features are what we 
# consider useful to the analysis. Categorical Variables (sex, cp, fbs, restecg, exang, slope
# and thal) are not considered useful. 

heart_data_continuous <- heart_data %>%
  select(age, trestbps, chol, thalach, oldpeak, ca)

head(heart_data_continuous, 10)

######################## SCALING #######################

# Before running K-means, scale the dataset to ensure that the variables are on a 
# comparable size, allowing for a fair and accurate cluster analysis.

heart_data_scaled <- scale(heart_data_continuous)

head(heart_data_scaled, 10)

########## To determine the Optimium Value of K ########################

# Now that we have concluded with scaling the dataset, we are able to run our cluster 
# analysis with K-means. However, we need to determine the optimal value of K for our
# cluster analsysis. There are different ways we can explore the optimial value of K and they
# include domain knowledge, elbow method, silhouette coefficient and gap statistics and we will
# be exploring them all.

# DOMAIN KNOWLEDGE : By default, we are running an unsupervised classification to determine 
# patients in 2 classes of no disease and patients with increased levels of heart disease, so 
# it would be ideal to use K value of 2.

# Elbow Method
wcss <- vector()
for (i in 1:10) {
  kmeans_model <- kmeans(heart_data_scaled, centers = i, nstart = 10)
  wcss[i] <- kmeans_model$tot.withinss
}
plot(1:10, wcss, type = "b", xlab = "Number of Clusters (K)", ylab = "WCSS")

# The result of the Elbow Method was not clear on what number of cluster would 
# result to the optimum value of K so, we continue to explore other methods.

# Silhouette Coefficient
silhouette_score <- function(k){
  km <- kmeans(heart_data_scaled, centers = k, nstart=25)
  ss <- silhouette(km$cluster, dist(heart_data_scaled))
  mean(ss[, 3])
}
k <- 2:10
avg_sil <- sapply(k, silhouette_score)
plot(k, type='b', avg_sil, xlab='Number of clusters', ylab='Average Silhouette Scores', frame=TRUE)

# The above method of calculating silhouette score using silhouette() and plotting
# the results states that optimal number of clusters as 2

# Gap statistic
fviz_nbclust(
  x = heart_data_scaled,
  FUN = kmeans,    # Type of clustering
  method = "gap_stat",
  nboot = 40,    # Number of start permutations
  k.max = 10,
  linecolor = "blue"
) +
  theme(
    panel.grid.major = element_line()    # Add gridlines for clarity
  )



# The result of Gap statistic shows that the optimal number of clusters is 2 and that 
# conforms with the result of silhuotte coeeficient and domain knowledge .

########## K-Means ####################

heart_cluster <- kmeans(heart_data_scaled, 2)
heart_cluster


######### Attaching the Cluster to the Original data ####################

heartClustered <- cbind(heart, "cluster" = heart_cluster$cluster)
heartClustered$cluster <- as.factor(heartClustered$cluster)
head(heartClustered)


#heartClustered %>% gather(attributes, value, 1:ncol(heartClustered)-1) %>%
#  ggplot(aes(x = value, group = cluster, fill = cluster)) +
#  geom_histogram(color = "black", alpha = 0.5) +
 # facet_wrap(~attributes, scales = "free")


########## CONFUSION MATRIX, ACCURACY, SENSITIVITY AND SPECIFICITY ##################
# K-means clustering is an unsupervised classification and most times do not come with ground 
# truth labels so we do not calculate check for the Confusion matrix. However, our data came with 
# ground-truth labels where the num column has 2 classifications, <50 maeans no disease
# and >50_1 means increased level of heart disease.

# Create tibbles for positive and negative
positive <- tibble(
  True = nrow(filter(heartClustered, num == ">50_1" & cluster == 2)),
  False = nrow(filter(heartClustered, num == ">50_1" & cluster == 1))
) %>% 
  t() %>% 
  as_tibble()%>% 
  set_names(c("Positive"))

negative <- tibble(
  True = nrow(filter(heartClustered, num == "<50" & cluster == 1)),
  False = nrow(filter(heartClustered, num == "<50" & cluster == 2))
)%>% 
  t() %>% 
  as_tibble()%>% 
  set_names(c("Negative"))

# Combine positive and negative tibbles
confusion_matrix <- data.frame(positive, negative)

row.names(confusion_matrix) <- c("TRUE", "FALSE")

# confusion matrix
print(confusion_matrix)

#ACCURACY 
paste("Accuracy = ", 
      round((confusion_matrix[1,1] + confusion_matrix[1,2])/ sum(confusion_matrix), 5))

# SENSITIVITY OF THE MODEL

paste("Sensitivity:", round(
  confusion_matrix[1,1]/ (confusion_matrix[1,1] + confusion_matrix[2,1]), 3
))

# SENSITIVITY OF THE MODEL

paste("Specificity = :", round(
  confusion_matrix[1,2]/ (confusion_matrix[1,2] + confusion_matrix[2,2]), 3
))

############# HIERARCHICAL CLUSTERING ###########################

# Hierarchical clustering is a valuable tool for discovering insights in your data and it 
# doesn't require the user to specify the number of clusters beforehand unlike K-means. The 
# idea behind hierarchical clustering is basically to show which set of sample are most similar
# to one another. For the purpose of this project, we will use the Euclidean distance
# for analysis which is the default distance. However, there are distances used for
# hierarchical clustering such as Manhattan distance and cosine similarity. 
library(dendextend)

distMat <- dist(heart_data_scaled, method = 'euclidean')
hcl <- hclust(distMat, method = "average")
hcl.ext <- as.dendrogram(hcl)
plot(colour_branches(hcl.ext, h = 5))

############
distMat <- dist(heart_data_scaled, method = 'euclidean')
hcl <- hclust(distMat, method = "single")
hcl.ext <- as.dendrogram(hcl)
plot(colour_branches(hcl.ext, main = "", xlab = "", ylab = "", axes = F, sub = "",
                     cex = 0.4))

############
distMat <- dist(heart_data_scaled, method = 'euclidean')
hcl1 <- hclust(distMat, method = "complete")
hcl1.ext <- as.dendrogram(hcl1)
plot(colour_branches(hcl1.ext, h = 2))

############
distMat <- dist(heart_data_scaled, method = 'euclidean')
hcl <- hclust(distMat, method = "ward.D2")
hcl.ext <- as.dendrogram(hcl)
plot(colour_branches(hcl.ext, h = 2))

############
distMat <- dist(heart_data_scaled, method = 'euclidean')
hcl <- hclust(distMat, method = "centroid")
hcl.ext <- as.dendrogram(hcl)
plot(colour_branches(hcl.ext, h = 2))
