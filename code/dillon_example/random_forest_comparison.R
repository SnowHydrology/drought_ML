# Script for exploring various machine learning strategies in R

# 1) randomForest package as used in NIDIS proposal
# 2) tidymodels
# 3) mlr3 (the updated version of mlr)

# Note 1: We do not include caret as Max Kuhn is currently a tidymodel dev

# Note 2: This is not a performance comparison. This is to evaluate ease of use
# and accessing 'under the hood' data.

# Keith Jennings
# kjennings@lynkertech.com
# 2021-01-19

# Load packages
library(tidyverse)
library(tidymodels)
library(mlr3)
library(randomForest)
library(here)
library(cowplot); theme_set(theme_cowplot())


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
  

################################################################################
###########################  2) randomForest  ##################################
################################################################################


# Prep data
dillon_inflow_pct_swe <- filter(indicator_impact_data, basin == "dillon") %>% 
  select(., inflow_pct, max_swe_in_av, amj_ppt, prev_ppt, jja_tair, spei_12, pdsi) %>% na.omit()

# Run randomForest 
set.seed(3002)
rf_inflow_pct_dillon_swe <-
  randomForest(inflow_pct ~ ., data = dillon_inflow_pct_swe)


################################################################################
###########################  3) tidymodels  ####################################
################################################################################

# Run tidymodels with randomForest engine
tm_inflow_pct_dillon_swe <-  rand_forest(trees = 500, mode = "regression") %>%
  set_engine("randomForest") %>%
  fit(inflow_pct ~ ., data = dillon_inflow_pct_swe)


