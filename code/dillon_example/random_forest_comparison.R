# Script for exploring various machine learning strategies in R

# 1) randomForest package as used in NIDIS proposal
# 2) tidymodels
# 3) mlr3 (the updated version of mlr)

# We do not include caret as Max Kuhn is currently a tidymodel dev

# Keith Jennings
# kjennings@lynkertech.com
# 2021-01-19

# Load packages
library(tidyverse)
library(mlr3)
library(randomForest)
library(here)
library(cowplot); theme_set(theme_cowplot())


################################################################################
############################  Import Data  #####################################
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
  

################################################################################################
#################################  Random Forest Models  #######################################
################################################################################################

################################################################################################
# Model set 1: Inflow pct of normal

#Prep data for SWE and no SWE runs
dillon_inflow_pct_swe <- filter(indicator_impact_data, basin == "dillon") %>% 
  select(., inflow_pct, max_swe_in_av, amj_ppt, prev_ppt, jja_tair, spei_12, pdsi) %>% na.omit()

#Run random forest with SWE
set.seed(3002)
rf_inflow_pct_dillon_swe <-
  randomForest(inflow_pct ~ ., data = dillon_inflow_pct_swe)
rf_inflow_pct_dillon_swe


################################################################################################
#############################  Data Extraction for Analysis  ###################################
################################################################################################

#Our models worked based to predict reservoir inflows at Dillon
#For the proposal, we'll plot variable importance scores
#And the distribution of Max SWE values predicting inflows < 75% of normal

#Extract variable importance scores
rf_inflow_pct_dillon_swe_VARIMPORT <- as.data.frame(rf_inflow_pct_dillon_swe[["importance"]])
rf_inflow_pct_dillon_swe_VARIMPORT$indicator <- row.names(rf_inflow_pct_dillon_swe_VARIMPORT)
rf_inflow_pct_dillon_swe_VARIMPORT$indicator <- factor(rf_inflow_pct_dillon_swe_VARIMPORT$indicator, 
                                                       levels = c("jja_tair", "prev_ppt", "pdsi", "amj_ppt", "spei_12", "max_swe_in_av"))

#Plot the importance scores
dillon_importance <- 
  ggplot(rf_inflow_pct_dillon_swe_VARIMPORT, aes(indicator, IncNodePurity, fill = IncNodePurity)) +
  geom_bar(stat = "identity") +
  coord_flip()+
  labs(x = "Indicator", y = "Importance")+
  scale_fill_viridis_c(option = "B", end = 0.8) +
  scale_x_discrete(labels = c("Summer T", "Prev. Yr. Ppt.", "PDSI", "Spring Ppt.", "SPEI", "Max SWE")) +
  theme(legend.position = "none")

#Extract the SWE distribution and compute means
rf_inflow_pct_dillon_swe_TREES <- data.frame()
for(i in 1:500){
  rf_inflow_pct_dillon_swe_TREES <- 
    bind_rows(rf_inflow_pct_dillon_swe_TREES ,randomForest::getTree(rfobj = rf_inflow_pct_dillon_swe, k = i, labelVar = TRUE))
}
colnames(rf_inflow_pct_dillon_swe_TREES) <- c("left_d", "right_d", "split_var", "split_pt", "status", "prediction")
rf_inflow_pct_dillon_swe_TREES_SWE <- filter(rf_inflow_pct_dillon_swe_TREES, split_var == "max_swe_in_av" & prediction < 75)

#Add identifier
rf_inflow_pct_dillon_swe_TREES_SWE$id <- "RF"
colnames(rf_inflow_pct_dillon_swe_TREES_SWE)[4] <- "max_swe_in_av"

#Bind to obs data
rf_inflow_pct_dillon_swe_TREES_SWE_analysis <- 
  dillon_inflow_pct_swe %>% dplyr::select(., max_swe_in_av) %>% mutate(id = "OBS") %>% bind_rows(., rf_inflow_pct_dillon_swe_TREES_SWE)

#Calculate means
swe_means <- plyr::ddply(rf_inflow_pct_dillon_swe_TREES_SWE_analysis, "id", summarise, mean_swe = mean(max_swe_in_av))

#Plot the distribution of max swe from obs and from predictions of inflow < 75% of normal

swe_distros <- 
  ggplot(rf_inflow_pct_dillon_swe_TREES_SWE_analysis, aes(max_swe_in_av * 25.4, color = id))+
  geom_density(lwd = 1) +
  geom_vline(data = swe_means, aes(xintercept = mean_swe * 25.4, color = id), linetype ="longdash", size = .8) +
  scale_color_manual(values = c("black", "royalblue"), name = "Data source", labels = c("Observations", "Random Forest")) +
  labs(x = "Maximum SWE (mm)", y = "Density") +
  theme(legend.position = c(0.65,0.8))

#Plot both
swe_importance <-
  plot_grid(
    dillon_importance,
    swe_distros,
    ncol = 2, align = "vh", labels = "AUTO"
  )

#Export
save_plot(swe_importance, 
          filename = "~/Lynker Technologies/Division 2 Proposals - Documents/190828 - NOAA NIDIS Grant/02 - Working Docs/Schematics & Figures/rf_import_swe_distro.pdf", 
          base_height = 6, base_width = 11)
