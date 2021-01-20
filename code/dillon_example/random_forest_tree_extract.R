# This script includes function for extracting tree data from 
# randomForest and tidymodels objects

# Requires tidyverse
library(tidyverse)

# Extract tree data from a randomForest object
# this is based on randomForest::getTree but that function only processes
# 1 tree at a time based on the value assigned to k
# This function extracts the whole forest
rf_getTree2 <- function(rfobj){
  # Extract all tree data as forest
  forest <- cbind(c(rfobj$forest$leftDaughter),
                c(rfobj$forest$rightDaughter),
                c(rfobj$forest$bestvar),
                c(rfobj$forest$xbestsplit),
                c(rfobj$forest$nodestatus),
                c(rfobj$forest$nodepred)) 
  
  # Add column names
  colnames(forest) <- c("left_d", "right_d", "split_var", "split_pt", "status", "prediction")
  
  # Convert forest to tibble
  forest <- forest %>% as_tibble()
  
  # Exclude all non-valid trees (status == 0)
  forest <- forest %>% 
    filter(status != 0)
  
  # Print forest
  forest
}


# Extract tree data from a tidymodels object
# this is based on randomForest::getTree as rf_getTree2 is above
tm_getTree2 <- function(tmobj){
  # Extract all tree data as forest
  forest <- cbind(c(tmobj$fit$forest$leftDaughter),
                  c(tmobj$fit$forest$rightDaughter),
                  c(tmobj$fit$forest$bestvar),
                  c(tmobj$fit$forest$xbestsplit),
                  c(tmobj$fit$forest$nodestatus),
                  c(tmobj$fit$forest$nodepred)) 
  
  # Add column names
  colnames(forest) <- c("left_d", "right_d", "split_var", "split_pt", "status", "prediction")
  
  # Convert forest to tibble
  forest <- forest %>% as_tibble()
  
  # Exclude all non-valid trees (status == 0)
  forest <- forest %>% 
    filter(status != 0)
  
  # Print forest
  forest
}

