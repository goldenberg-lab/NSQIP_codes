# ---- 'Reproducing' Maizlin et al. 2014 ----
# Uses lasso logistic regression instead of stepwise logistic regression for variable selection.
# - 5 fold cross validation for each of the 5 outcomes
# - uses lambda 1 standard deviation of the lowest cv error
# - calculates and visualizes pr/rc and auroc


library(tidyverse)
library(naniar)
library(glmnet)
library(tictoc)
library(patchwork)


# ---- read in the x matrix and y -----
maizlin_x <- read_csv("output/maizlin_x.csv")
maizlin_y <- read_csv("output/maizlin_y.csv")

# sanity checks
nrow(maizlin_x) == 183233
nrow(maizlin_y) == nrow(maizlin_x)

miss_var_summary(maizlin_x)
miss_var_summary(maizlin_y)
# should be no missingness


#  ---- randomly subset from the data for testing ----
# set.seed(1)
# 
# testing_rows <- sample.int(nrow(maizlin_x),  # sample all rows 50%
#                 nrow(maizlin_x) * 0.2,
#                 replace = FALSE)
# maizlin_x_ss <- maizlin_x[testing_rows,]
# maizlin_y_ss <- maizlin_y[testing_rows,]
# 
# dim(maizlin_x_ss)
# dim(maizlin_y_ss)
# 
# maizlin_x <- maizlin_x_ss
# maizlin_y <- maizlin_y_ss

# --- splitting the data ----
# follow a 50% training/validation scheme 
set.seed(1)
split = 0.5
train_rows = sample.int(nrow(maizlin_x),  # sample all rows 50% 
                        nrow(maizlin_x) * split, 
                        replace = FALSE)
# length(train_rows)

train_x = as.matrix(maizlin_x[train_rows, -c(1:2)])
test_x = as.matrix(maizlin_x[-train_rows, -c(1:2)])


dim(train_x); dim(test_x)
# length(train_y); length(test_y)

# we will iterate through the outcomes 
maizlin_res <- names(maizlin_y)[-c(1:2)] %>%
  map(function(this_outcome){
    
    print(this_outcome)
    
    train_y = maizlin_y[train_rows, -c(1:2)][[this_outcome]]
    test_y = maizlin_y[-train_rows, -c(1:2)][[this_outcome]]
    
    # print(train_y)
    # print(test_y)
    
    # check for balancedness
    table(train_y)
    table(test_y)
    
    set.seed(1) # for reproducibility
    tic()
    lasso_logis_cv <- cv.glmnet(x = train_x, 
                                y = train_y,
                                nfolds = 5, 
                                family = "binomial", 
                                alpha = 1,
                                type.measure = "auc")
    toc()
    
    plot(lasso_logis_cv)
    
    # lasso_logis_cv$lambda
    # lasso_logis_cv$lambda.min

    fitted_probs <- predict(lasso_logis_cv, 
                            newx= train_x,
                            type = "response",
                            s = lasso_logis_cv$lambda.1se)[,1]
    
    predicted_probs <- predict(lasso_logis_cv, 
                               newx= test_x,
                               type = "response",
                               s = lasso_logis_cv$lambda.1se)[,1]
    
    fitted_probs %>% histogram()
    predicted_probs %>% histogram()
    
    # roc( test_y, predicted_probs) # 0.8136
    # roc( train_y, fitted_probs) # 0.8255
    
    # 
    # roc_res <- roc_func(lasso_logis_cv, type = "response", title = "Organ Space SSI Classifier")
    # 
    # 
    # lasso_logis_cv
    # 
    # var_imp_p <- coef(lasso_logis_cv, s = lasso_logis_cv$lambda.1se) %>%
    #   broom::tidy() %>%
    #   as_tibble() %>%
    #   mutate(value = abs(value)) %>%
    #   filter(value != 0) %>%
    #   ggplot(aes(x = reorder(row, value), y = value, fill = value)) +
    #   geom_col(colour = "black", alpha = 0.9) +
    #   coord_flip() +
    #   scale_fill_viridis_c() +
    #   guides(fill = FALSE)  +
    #   labs(y = "Coefficients", x = "Predictors", title = "LASSO Logistic Regression Final Coefficients")
    # 
    # print(var_imp_p)
    # 
    roc_res <- tibble(set = c("training", "validation"))
    roc_res$outcome <- this_outcome
    roc_res$actual_y <- c()
    roc_res$actual_y[roc_res$set == "training"] <- list(train_y)
    roc_res$actual_y[roc_res$set == "validation"] <- list(test_y)
    roc_res$model_y[roc_res$set == "training"] <- list(fitted_probs)
    roc_res$model_y[roc_res$set == "validation"] <- list(predicted_probs)
    roc_res$caseid[roc_res$set == "training"] <- list(maizlin_y[train_rows,][["caseid"]])
    roc_res$caseid[roc_res$set == "validation"] <- list(maizlin_y[-train_rows,][["caseid"]])
    
    return(roc_res)
    

  })

maizlin_res_df <- bind_rows(maizlin_res) %>% unnest(actual_y, model_y, caseid) 

# ---- save down csv ----
# write_csv(maizlin_res_df, "output/maizlin_res_df.csv")
# ----- read in csv ----
# maizlin_res_df <- read_csv("output/maizlin_res_df.csv")

# distribution of the outcome for probabilities for each outcome
png(filename = paste0(figure_dir, "maizlin_probs_distribution.png"), width = 840, height = 640)
maizlin_res_df %>% 
  ggplot(aes(x = model_y)) +
  geom_histogram() +
  facet_wrap(set ~ outcome, scales = "free", nrow = 2)
dev.off()


# ---- compute pr/rc and auroc ----
# by_steps = 0.001
pr_rc_df <- unique(maizlin_res_df$outcome) %>%
  map(function(this_outcome){
    print(this_outcome)
    
    # train_y <- maizlin_res_df[maizlin_res_df$outcome == this_outcome & maizlin_res_df$set == "training", ]$actual_y
    # fitted_y <- maizlin_res_df[maizlin_res_df$outcome == this_outcome & maizlin_res_df$set == "training", ]$model_y
    
    test_y <- maizlin_res_df[maizlin_res_df$outcome == this_outcome & maizlin_res_df$set == "validation", ]$actual_y
    pred_y <- maizlin_res_df[maizlin_res_df$outcome == this_outcome & maizlin_res_df$set == "validation", ]$model_y
    
   
    # print(paste0("Max Prob of Test: ", max(test_y)))
    print(paste0("Max Prob of yhats: ", max(pred_y)))
    
    # from 0 to the max value of predicted y
    max(pred_y)
    steps <- seq(min(pred_y), max(pred_y), by = 0.001)
    
    threshold <- 0.0 # tried using 0.5 but then got no predicted values in the positive class
                     # that's because the probabilities don't span the range of 0 - 1 (oops)
    
    # at various thresholds, compute pr/rc
   pr_rc_res <-  1:length(steps) %>%
      
      map_dfr(function(idx){
        threshold = threshold + steps[idx]
        
        print(threshold)
        test_y <- as.numeric(test_y > threshold)
        pred_y <- as.numeric(pred_y > threshold)
        
        conf_mat <- as.matrix(table(pred_y, test_y))
        
        # print(conf_mat)
        
        fp <- conf_mat[2, 1] # false positives 
        tp <- conf_mat[2, 2] # true positives
        
        # precision = true positives/true + false positives, the proportion of positive cases correctly called out of all positive cases
        
        (pr = tp/(tp + fp))
        
        # recall= true positives/(), proportion of those classified as true positive cases among those who are positive 
        # in other words, the ability o f the classifier to detect the positive event
        fn <- conf_mat[1, 2] # false negs, positives called as negatives
        (rc <- tp/(tp + fn))
        
        # specificity , the probability of a negative case, given the case is negative
        tn <- conf_mat[1, 1]
        sp <- tn/(tn + fp)
        sp
        
        res <- tibble(this_outcome, threshold, pr, rc, sp )
        
        }
      )

    return(pr_rc_res)
  })


# join variable description 
pr_rc_df <- pr_rc_df %>% 
  bind_rows() %>%
  left_join(master_key, by = c("this_outcome" = "vars"))


# ---- visualizing AUROC and PR-RC -----

png(filename = paste0(figure_dir, "maizlin_prrc_auroc.png"), width = 480, height = 840)

pr_rc_df  %>%
  ggplot(aes(y = pr, x = rc, colour = variable_label), alpha = 0.8) +
  geom_line(size = 1, alpha = 0.8) +
  # facet_wrap(~ this_outcome, scales = "free") +
  # scale_y_continuous(limits = c(0, 1)) +
  scale_color_brewer("Outcomes", palette = "Set1") +
  # theme_minimal(base_size = 14) +
  ggtitle("Precision-Recall Curves", "Maizlin et al. 2017 Outcomes")  + 
  theme(legend.direction = "vertical",
        legend.position = "top") +

pr_rc_df %>%
  ggplot(aes(y = rc, x = (1 - sp), colour = variable_label), alpha = 0.8) +
  geom_line(size = 1, alpha = 0.8) +
  geom_abline(aes(intercept = 0, slope = 1), linetype = "dotted") + 
  # facet_wrap(~ this_outcome, scales = "free") +
  # scale_y_continuous(limits = c(0, 1)) +
  scale_color_brewer("Outcomes", palette = "Set1") +
  # theme_minimal(base_size = 14)  +
  ggtitle("AUROC", "Maizlin et al. 2017 Outcomes") + 
  guides(colour = FALSE) + 
  plot_layout(nrow = 2)

dev.off()


