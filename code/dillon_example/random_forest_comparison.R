# Script for exploring various machine learning strategies in R

# 1) randomForest package as used in NIDIS proposal
# 2) tidymodels

# Note 1: We do not include caret as Max Kuhn is currently a tidymodel dev

# Note 2: This is not a performance comparison. This is to evaluate ease of use
# and accessing 'under the hood' data.

# Note 3: May also consider mlr3 and its slew of associated packages

# Keith Jennings
# kjennings@lynkertech.com
# 2021-01-19

# Load packages
library(tidyverse)
library(tidymodels)
library(randomForest)
library(here)
library(cowplot); theme_set(theme_cowplot())

# Source the random_forest_tree_extract.R script for getTree functions
source(here("code", "dillon_example", "random_forest_tree_extract.R"))

################################################################################
########################### 1)  Import Data  ###################################
################################################################################

# Use here package to locate RDS files 
# These were processed for an example analysis in the NIDIS proposal
filefolder <- here("data", "dillon_example")
files <- dir(filefolder, pattern = "*.RDS")

# Import all the RDS files into a tibble
# map applies the readRDS function to all files
# reduce converts the list into a single tibble using full_join
indicator_impact_data <- paste(filefolder, files, sep = "/") %>%
  map(readRDS) %>% 
  reduce(full_join, by = c("year", "basin")) %>% 
  filter(basin == "dillon") # remove unnecessary barker data
  
# Prep data for the ML models
# Note: many have na methods that don't require removing NA values a priori
dillon_inflow_pct_swe <- filter(indicator_impact_data, basin == "dillon") %>% 
  ungroup() %>% # needed to get rid of "basin" grouping variable
  select(inflow_pct, max_swe_in_av, amj_ppt, prev_ppt, jja_tair, spei_12, pdsi) %>% 
  na.omit()

################################################################################
###########################  2) randomForest  ##################################
################################################################################

# Set seed for reproducibility
set.seed(3002)

# Run the model and save in a randomForest object
rf_inflow_pct_dillon_swe <-
  randomForest(inflow_pct ~ ., data = dillon_inflow_pct_swe)

# Extract variable importance scores
rf_inflow_pct_dillon_swe_VARIMPORT <- as_tibble(rf_inflow_pct_dillon_swe$importance) %>% 
  mutate(indicator = names(rf_inflow_pct_dillon_swe$forest$ncat))

# Extract forest data
rf_inflow_pct_dillon_swe_FOREST <- rf_getTree2(rf_inflow_pct_dillon_swe)

################################################################################
###########################  3) tidymodels  ####################################
################################################################################

# Run tidymodels with randomForest engine
tm_inflow_pct_dillon_swe <-  rand_forest(trees = 500, mode = "regression") %>%
  set_engine("randomForest") %>%
  fit(inflow_pct ~ ., data = dillon_inflow_pct_swe)

# Extract variable importance scores
tm_inflow_pct_dillon_swe_VARIMPORT <- as_tibble(tm_inflow_pct_dillon_swe$fit$importance) %>% 
  mutate(indicator = names(tm_inflow_pct_dillon_swe$fit$forest$ncat))

# Extract forest data
tm_inflow_pct_dillon_swe_FOREST <- tm_getTree2(tm_inflow_pct_dillon_swe)


################################################################################
####################  4) Compare Extracted Output  #########################
################################################################################

