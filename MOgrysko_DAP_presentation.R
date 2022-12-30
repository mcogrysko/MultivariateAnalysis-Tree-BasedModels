library(tree)
setwd("/Users/mogrysko/Documents/Coursework/DataScience/MA/DAP")
data <- read.table("DATA.txt", header=TRUE, sep=";");data
#plot for presentation
hist(data$GRADE, main="Histogram of Student Grades", xlab="Student Grades")
SUCCESS <- ifelse(data$GRADE <= 3, "No", "Yes")
#plot for presentation
plot(factor(SUCCESS), main="Histogram of Student Success", xlab="Student Success")
tmpdata <-data[,2:31]
colnames(tmpdata) <- c("age","sex","hs_type","scholar_type","add_work","activities","partner","salary","transportation","accommodation","m_edu","f_edu","num_sibling","parent_status","m_occ","f_occ","study_hrs","nonsci_read","sci_read","attend_seminar","proj_impact","attend_class","prep_midterm1","prep_midterm2","class_notes","listen_class","class_disc","flip_class","cum_gpa_sem","cum_gpa_grad")
sdata <- data.frame(sdata <- tmpdata, SUCCESS)
sdata$age <- as.factor(sdata$age);sdata$sex <- as.factor(sdata$sex);sdata$hs_type <- as.factor(sdata$hs_type);sdata$scholar_type <- as.factor(sdata$scholar_type);sdata$add_work <- as.factor(sdata$add_work);sdata$activities <- as.factor(sdata$activities);sdata$partner <- as.factor(sdata$partner);sdata$salary <- as.factor(sdata$salary);sdata$transportation <- as.factor(sdata$transportation);sdata$accommodation <- as.factor(sdata$accommodation);sdata$m_edu <- as.factor(sdata$m_edu);sdata$f_edu <- as.factor(sdata$f_edu);sdata$num_sibling <- as.factor(sdata$num_sibling);sdata$parent_status <- as.factor(sdata$parent_status);sdata$m_occ <- as.factor(sdata$m_occ);sdata$f_occ <- as.factor(sdata$f_occ);sdata$study_hrs <- as.factor(sdata$study_hrs);sdata$nonsci_read  <- as.factor(sdata$nonsci_read);sdata$sci_read <- as.factor(sdata$sci_read);sdata$attend_seminar  <- as.factor(sdata$attend_seminar);sdata$proj_impact <- as.factor(sdata$proj_impact);sdata$attend_class <- as.factor(sdata$attend_class);sdata$prep_midterm1 <- as.factor(sdata$prep_midterm1);sdata$prep_midterm2 <- as.factor(sdata$prep_midterm2);sdata$class_notes <- as.factor(sdata$class_notes);sdata$listen_class <- as.factor(sdata$listen_class);sdata$class_disc <- as.factor(sdata$class_disc);sdata$flip_class  <- as.factor(sdata$flip_class);sdata$cum_gpa_sem <- as.factor(sdata$cum_gpa_sem);sdata$cum_gpa_grad <- as.factor(sdata$cum_gpa_grad);sdata$SUCCESS <- as.factor(sdata$SUCCESS)
# Full model
tree.sdata.full <- tree(SUCCESS ~ ., split="deviance", data=sdata)
summary(tree.sdata.full)
tree.sdata.full <- tree(SUCCESS ~ ., split="deviance", data=sdata)
summary(tree.sdata.full)
dev.new(height=800, width=1200)
plot(tree.sdata.full)
title(main="Classification Tree (Deviance) - Full Data Set")
text(tree.sdata.full, cex=0.75, pretty=0)
# Model Evaluation
set.seed(101)
# 70/30 train/test split
train <- sample(1:nrow(sdata), size=101, replace=FALSE)
# Training set
sdata.train <- sdata[train,]
# Test set
sdata.test <- sdata[-train,]
# Fit training data
tree.sdata  <- tree(SUCCESS ~ ., split="deviance", data=sdata.train)
summary(tree.sdata)
tree.pred <- predict(tree.sdata, sdata.test[-31], type="class")
table(tree.pred); table(sdata.test$SUCCESS)
table(tree.pred, sdata.test$SUCCESS)
mean(tree.pred != sdata.test$SUCCESS)     # validation error rate

#plot training tree
dev.new(height=800, width=1200)
plot(tree.sdata)
title(main="Classification Tree (Deviance) - Training Data Set")
text(tree.sdata, cex=0.75, pretty=0)

tree.sdata.rpart <- rpart(SUCCESS ~ ., data=sdata.train)
rpart.plot(tree.sdata.rpart, extra=4)
vip(tree.sdata.rpart)

tree.sdata

# Pruning
set.seed(729)
# K-fold cross-validation
cv.sdata <- cv.tree(tree.sdata, FUN=prune.misclass)
# Plot Deviance and Number of Terminal Nodes
dev.new(height=800, width=800)
par(mfrow=c(1,2))
plot(cv.sdata$size, cv.sdata$dev, xlab="Number of terminal nodes", ylab="Deviance", type="b")
plot(cv.sdata$k, cv.sdata$dev, xlab="Cost-complexity pruning parameter", ylab="Deviance", type="b")

# Option 1 - 7 Terminal Nodes
prune.sdata.7 <- prune.misclass(tree.sdata, best=7)
summary(prune.sdata.7)

dev.new(height=800, width=1600)
plot(prune.sdata.7)
title(main="Classification Tree After Pruning - 7 Terminal Nodes")
text(prune.sdata.7, cex=0.75, pretty=0)

tree.pred.7 <- predict(prune.sdata.7, sdata.test, type="class")
table(tree.pred.7); table(sdata.test$SUCCESS)
table(tree.pred.7, sdata.test$SUCCESS)
mean(tree.pred.7 != sdata.test$SUCCESS)

prune.sdata.7$frame

# Option 2 - 9 Terminal Nodes
prune.sdata.9 <- prune.misclass(tree.sdata, best=9)
summary(prune.sdata.9)

dev.new(height=800, width=1200)
plot(prune.sdata.9)
title(main="Classification Tree After Pruning - 9 Terminal Nodes")
text(prune.sdata.9, cex=0.75, pretty=0)

tree.pred.9 <- predict(prune.sdata.9, sdata.test, type="class")
table(tree.pred.9); table(sdata.test$SUCCESS)
table(tree.pred.9, sdata.test$SUCCESS)
mean(tree.pred.9 != sdata.test$SUCCESS)


dim(sdata.test[-31])

# RF
library(randomForest)
set.seed(909)
sdata.rf = randomForest(SUCCESS ~ ., data=sdata.train, ntree=500, importance=TRUE)
sdata.rf
# mtry num splits
oob.values <- vector(length=14)
for(i in 1:14) {
  temp.sdata.rf <- randomForest(SUCCESS ~ ., data=sdata.train, mtry=i, ntree=500)
  oob.values[i] <- temp.sdata.rf$err.rate[nrow(temp.sdata.rf$err.rate),1]
}
oob.values
#redo with 4 as mtry
sdata.rf = randomForest(SUCCESS ~ ., data=sdata.train, ntree=500, mtry=4, importance=TRUE)
sdata.rf
#prediction
pred.test <- predict(sdata.rf,sdata.test)
table(observed=sdata.test$SUCCESS,predicted=pred.test)
#validation error rate
mean(pred.test != sdata.test$SUCCESS)








