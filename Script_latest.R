#Instacart Problem

#Remove Existing variables
rm(list=ls())


#Loading dependencies

library(dplyr)
library(ggplot2)
library(data.table)
library(caret)
library(modeest)
library(MLmetrics)


#Set current working directory
setwd("C:/Users/Aravind Atreya/Desktop/Kaggle/Instacart")

#Importing files
aisles <- fread("aisles.csv") 
departments <- fread("departments.csv")
order_products_prior <- fread("order_products__prior.csv")
order_products_train <- fread("order_products__train.csv")
orders <- fread("orders.csv")
products <- fread("products.csv")
sample_submission <- fread("sample_submission.csv")


#Creating a Product Table
products_all <- products %>%
  inner_join(aisles) %>%
  inner_join(departments) %>%
  select(-aisle_id,-department_id)

orders %>% filter(eval_set=="prior") %>% select(user_id) %>% unique() %>% nrow() #Total users
orders %>% filter(eval_set=="train") %>% select(user_id) %>% unique() %>% nrow() #Total train users
orders %>% filter(eval_set=="test") %>% select(user_id) %>% unique() %>% nrow() #Total test users

products_orders <- orders %>%
  inner_join(order_products_prior)

#Product Level
prd_metrics <- products_orders %>%
  group_by(product_id)%>%
  summarise(prod_num_orders=n(),
            prod_num_reorders=sum(reordered,na.rm=T))%>%
  mutate(prod_reorder_ratio=prod_num_reorders/prod_num_orders)%>%
  select(-prod_num_orders,-prod_num_reorders) %>%
  arrange(desc(prod_reorder_ratio))

head(prd_metrics)
#Most reordered product is product number 6433
products_all[which(products_all$product_id==6433),]
#Looks like Raw veggie wrappers is the most reordered product

#User level Metrics
user_metrics_1 <- orders%>%
  filter(eval_set=="prior")%>%
  group_by(user_id)%>%
  summarize(
    mean_order_interval= mean(days_since_prior_order,na.rm=T),
    prev_order_to_first_order=sum(days_since_prior_order,na.rm=T)
    #,most_ordered_day= mlv(order_dow,method="mfv")[['M']] #not working check later
  )

user_metrics_2 <- products_orders %>%
  filter(eval_set=="prior")%>%
  group_by(user_id)%>%
  summarize(
    num_orders=max(order_number),
    num_distinct_products=n_distinct(product_id)
  )
user_metrics_3 <-products_orders %>%
                 group_by(user_id,order_id)%>%
                 summarise(basket_size=max(order_number))%>%
                 group_by(user_id)%>%
                 summarise(avg_basket_size=mean(basket_size))

user_metrics_4 <- orders %>%
                  filter(eval_set!="prior")%>%
                  select(user_id,order_id,eval_set,
                  days_since_prev_order=days_since_prior_order)

user_metrics <- user_metrics_1 %>%
                inner_join(user_metrics_2) %>%
                inner_join(user_metrics_3) %>%
                inner_join(user_metrics_4)

head(user_metrics)
dim(user_metrics)

#Data, Training and Testing

#Identifying userids for train orders
order_products_train$user_id <- orders$user_id[match(order_products_train$order_id,orders$order_id)]

data_total <- products_orders %>%
              group_by(user_id,product_id)%>%
              summarize(
              prd_num_orders = n(),
              first_order=min(order_number),
              last_order=max(order_number),
              mean_basket_position=mean(add_to_cart_order)
            )
data_total <- data_total %>%
              inner_join(prd_metrics) %>%
              inner_join(user_metrics)

data_total <- data_total %>%
              left_join(order_products_train %>% 
                        select(user_id,product_id,reordered))
#Train

data_train <- data_total %>%
              filter(eval_set=="train")
data_train$eval_set <- NULL
data_train$user_id <- NULL
data_train$product_id <- NULL
data_train$order_id <- NULL
data_train$reordered[is.na(data_train$reordered)] <-0 

#Test
data_test <- data_total %>% 
             filter(eval_set=="test")
data_test$eval_set <- NULL
data_test$user_id <-NULL
test_product_id <- data_test$product_id
data_test$product_id <- NULL
test_order_id <- data_test$order_id
data_test$order_id <- NULL
data_test$reordered <- NULL

#sample random 20% of the train dataset
set.seed(3)
sample_rows <- sample(nrow(data_train)*0.5)
sample_train <- data_train[sample_rows,]
sample_train$reordered <- as.factor(sample_train$reordered)
levels(sample_train$reordered) <- c("No","Yes")

#Using custom metric F1
set.seed(346)
dat <- twoClassSim(200)
f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- F1_Score(y_pred = data$pred, y_true = data$obs, positive = lev[1])
  c(F1 = f1_val)
}

#Using caret to find a model
set.seed(4)
fitControl  <- trainControl(method = "cv",
                            number = 5,
                            repeats =3,
                            summaryFunction = f1,
                            classProbs = TRUE,
                            verboseIter = TRUE)


set.seed(5)
training <- train(reordered~.
                  ,data=sample_train
                  ,method="gbm"
                  ,trControl =fitControl
                  ,metric = "F1")
#Dividing test in to 5 parts and prediciting, assize of test exceeds some limit
split <- seq(0,nrow(data_test),nrow(data_test)/4)
test_1 <- data_test[1:split[2],]
test_2 <- data_test[(split[2]+1):split[3],]
test_3 <- data_test[(split[3]+1):split[4],]
test_4 <- data_test[(split[4]+1):split[5],]
pred_1 <- predict(training,test_1)
pred_1 <- data.frame(pred_1)
colnames(pred_1) <- "Reordered"
pred_2 <- predict(training,test_2)
pred_2 <- data.frame(pred_2)
colnames(pred_2) <- "Reordered"
pred_3 <- predict(training,test_3)
pred_3 <- data.frame(pred_3)
colnames(pred_3) <- "Reordered"
pred_4 <- predict(training,test_4)
pred_4 <- data.frame(pred_4)
colnames(pred_4) <- "Reordered"
pred <- rbind(pred_1,pred_2,pred_3,pred_4)
nrow(data_test)==nrow(pred)

#Sample Submission Preparataion
submission <- data.frame(order_id=test_order_id,
                         product_id=test_product_id,
                         pred)

submission_1 <- submission %>%
                filter(Reordered=="Yes")%>%
                 group_by(order_id) %>%
                 summarize(products= paste(product_id,collapse= " "))
none_order_ids <- unique(test_order_id[!(test_order_id %in% submission_1$order_id)])
submission_2 <- data.frame(order_id=none_order_ids,products="None")
sample_submission <- rbind(submission_1,submission_2)%>%
                     arrange(desc(order_id))
write.csv(sample_submission,"Sample_Submission.csv",row.names = F)

