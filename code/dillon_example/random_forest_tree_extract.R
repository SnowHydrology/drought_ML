# This script includes function for extracting tree data from 
# randomForest and tidymodels objects

rf_getTree2 <- function(rfobj){
  tree <- cbind(c(rfobj$forest$nodestatus),
                c(rfobj$forest$leftDaughter),
                c(rfobj$forest$rightDaughter))
  tree
}
