# Classification of wildfire presence/absence at HUC6 level

# Load packages
library(tidymodels)
library(tidyverse)
library(cowplot); theme_set(theme_cowplot())
library(doMC); registerDoMC(cores = 4) # for parallel model fitting & tuning
library(vip) # variable importance plots
library(rgdal)
library(sp)
library(leaflet)
# also need raster, but call the package in line to prevent loss of tidyverse functionality


# Load data
data_dir = "data/wildfire/"
df <- readRDS(paste0(data_dir, "all_data.RDS")) %>% ungroup() 

# Select a few variables for the model
# Future runs will include more, but testing is quicker this way
df <- select(df, year, huc10, FracBurnedArea, summer_pr, summer_tmmx, summer_pet,
             summer_erc, summer_bi, summer_fm1000) 

# Add a huc 6 column based on the HUC10 code
df <- df %>% 
  mutate(huc6 = str_sub(huc10, 1, 6))

# Summarize by huc6 and add a fire presence/absence column
df <- select(df, -huc10) %>% 
  group_by(huc6, year) %>% 
  summarize_all(.funs = mean) %>% 
  ungroup %>% 
  mutate(fire = as.factor(ifelse(FracBurnedArea > 0,
                                 "yes", "no"))) %>% 
  na.omit()  # There is a step_naomit, but it breaks the model fit when there are 
             # too many missing values in the analysis/assessment sets

# Set seed so that the analysis is reproducible
set.seed(6547)

# Split the data into training and testing
df_split <- initial_split(data = df,
                          prop = 0.75,  # This is the proportion of data allocated to training
                          strata = "fire")  # Stratify the sampling based on this variable

# Make new data frames of the training and testing data
df_train <- training(df_split)
df_test <- testing(df_split)


# Make a recipe
df_recipe <- recipe(fire ~ ., data = df_train) %>%
  update_role(year, FracBurnedArea, huc6, new_role = "analysis")

# Create some folds
df_folds <- vfold_cv(df_train, v = 10)

# Define the random forest model
rf_mod <- rand_forest() %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification") %>% 
  set_args(mtry = tune(),
           trees = tune())

# Make a workflow
rf_flow <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(df_recipe)


# specify which values eant to try
rf_grid <- expand.grid(mtry = c(3, 4, 5), 
                       trees = c(100, 300, 500))

# Run the tuning grid
rf_tune_results <- rf_flow %>%
  tune_grid(resamples = df_folds, #CV object
            grid = rf_grid, # grid of values to try
            metrics = metric_set(accuracy, roc_auc) # metrics we care about
  )

# print results
rf_tune_results %>%
  collect_metrics() %>% 
  knitr::kable()

# Extract the best model parameters
param_best <- rf_tune_results %>% 
  select_best(metric = "accuracy") # can also choose "roc_auc"
 
# Add this to the workflow
rf_flow_tuned <- rf_flow %>% 
  finalize_workflow(param_best)

# Evaluate the model by fitting to training and analyzing test
rf_fit <- rf_flow_tuned %>%
  # fit on the training set and evaluate on test set
  last_fit(df_split)

# Examine the model metrics
rf_fit %>% collect_metrics() %>% 
  knitr::kable()

# Create a confusion matrix
rf_fit %>% collect_predictions() %>% 
  conf_mat(truth = fire, estimate = .pred_class) %>% 
  autoplot(type = "heatmap", )

# To create the final version of the model, run the fit on the full dataset
# We will use this model to make future predictions
final_model <- rf_flow_tuned %>% 
  fit(df)

###############################################################################

# Extract the final fitted data
# This will be used for analysis, plotting, etc.
final_fit <- final_model %>% pull_workflow_fit()

# Add the predictions to the data
df_preds <- df %>% 
  mutate(pred_yes = final_fit$fit$predictions[,2],
         fire_pred = ifelse(pred_yes > 0.5, "yes", "no") %>% as.factor())

# Plot predicted fire / no fire by year
ggplot(df_preds, aes(year, fill = fire_pred)) +
  geom_bar(position = "dodge")

# Plot obs vs. sim fires
df_preds %>% 
  group_by(year) %>% 
  summarise(obs = sum(fire == "yes", na.rm = T),
            sim = sum(fire_pred == "yes", na.rm = T)) %>% 
  ggplot(aes(obs, sim)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0)

# Plot obs vs. sim fires by year
df_preds %>% 
  group_by(year) %>% 
  summarise(obs = sum(fire == "yes"),
            sim = sum(fire_pred == "yes")) %>% 
  pivot_longer(cols = c(obs, sim), names_to = "cat", values_to = "n_fires") %>% 
  ggplot(aes(year, n_fires, color = cat)) +
  geom_point() +
  geom_line(lwd = 1)


# Import HUC06 map
basins <- readOGR(dsn = "data/wildfire/", layer = "huc06_basins")

# Create an extent object for cropping basins
analysis_extent <- raster::extent(-115.429, -100.677,
                                  29.83, 46.4)

# Crop by extent 
basins_cropped <- raster::crop(basins, analysis_extent)

# Join 2012 wildfire data for viz
basins_2012 <- sp::merge(basins_cropped,
                         filter(df_preds, year == 2012) %>% select(HUC6 = huc6, fire_pred),
                         by = "HUC6")

# Make a color ramp for the map using viridis
map_pal <- colorFactor("magma", basins_2012$fire_pred)

# Make interactive map with leaflet
leaflet(basins_2012) %>% #make leaflet object with test2 data
  addTiles() %>% #add a basemap (default = open street map)
  addPolygons(popup = paste0(basins_2012$HUC6, "<br>", 
                             basins_2012$fire_pred), #popup county name and ead on click
              fillColor = ~map_pal(fire_pred), #fill with palette
              color = "black", #black outlines for counties
              fillOpacity = 0.6) %>% #semi-opaque fill
  addLegend(position = "bottomright", #add legend to bottom right
            pal = map_pal, #use color palette for legend
            values = ~fire_pred, #values are ead
            title = "Fire Predicted")


###############################################################################


# Run the cross-validation on analysis and assessment data
# rf_fit <- 
#   rf_flow %>% 
#   fit_resamples(df_folds, 
#                 control = control_resamples(save_pred = TRUE, verbose = TRUE))

# Run just on training
rf_fit <- rf_flow %>% fit(df_train)

rf_obj <- pull_workflow_fit(rf_fit)$fit


rf_test <- predict(rf_fit, new_data = df_test)
rf_test <- bind_cols(rf_test, na.omit(df_test))
rf_test$fire <- as.factor(rf_test$fire)

ggplot(rf_test, aes(FracBurnedArea, .pred)) + geom_point()

rmse(rf_test, truth = FracBurnedArea, estimate = .pred)
rsq(rf_test, truth = FracBurnedArea, estimate = .pred)
accuracy(rf_test, truth = fire, estimate = .pred_class)
detection_prevalence(rf_test, truth = fire, estimate = .pred_class)
npv(rf_test, truth = fire, estimate = .pred_class)
ppv(rf_test, truth = fire, estimate = .pred_class)
conf_mat(rf_test, truth = fire, estimate = .pred_class)

rf_obj %>% vip(geom = "point")


rf_varimport <- as_tibble(rf_fit$fit$) %>% 
  mutate(indicator = names(rf_varimport$fit$forest$ncat))


test <- rand_forest(trees = 500, mode = "classification") %>%
  set_engine("randomForest") %>%
  fit(fire ~ ., data = df)
