Machine Learning with Random Forest Models and the Tidymodels
================
Keith Jennings

## Why tidymodels?

Much like how the tidyverse is a group of packages for exploring,
modifying, and plotting data, tidymodels provides a unified framework
for statistical modeling in R. Its function-based approach allows for
intuitive model set up while also letting the user make any necessary
edits without changing the entire script.

## Why random forest models?

Random forest is a machine learning approach where the models learn
patterns from the data to make outcome predictions. It is a form of
supervised regression that can be easily run numerous times.

Assumptions + simplicity.

## Model setup

### Packages and data

A key part to any machine learning effort is the proper set up,
training, and validation of the included models. tidymodels makes this
process straightforward and reproducible. Conveniently, many of the same
bits of code can be used, even when the model engine changes.

First, let’s load the packages we need:

``` r
library(tidyverse)
library(tidymodels)
library(cowplot); theme_set(theme_cowplot()) # I like the cowplot because it makes plot pretty
```

Next, we’ll need to load some data:

Look at the data

### Prep the data

The first thing you need to do is split the data into *training* and
*testing* sets. We’ll use the former to optimize the random forest
models and the latter to independently test their efficacy. Here, we’ll
use functions from the `rsample` package within `tidymodels`.

## Acknowledgments

This was adapted from the following excellent articles:
-<https://juliasilge.com/blog/intro-tidymodels/>
-<https://www.brodrigues.co/blog/2018-11-25-tidy_cv/>
-<https://hansjoerg.me/2020/02/09/tidymodels-for-machine-learning/#tuning-model-parameters-tune-and-dials>

``` r
summary(cars)
```

    ##      speed           dist       
    ##  Min.   : 4.0   Min.   :  2.00  
    ##  1st Qu.:12.0   1st Qu.: 26.00  
    ##  Median :15.0   Median : 36.00  
    ##  Mean   :15.4   Mean   : 42.98  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00  
    ##  Max.   :25.0   Max.   :120.00

``` r
# Compute new variable
cars <- cars %>% 
  mutate(speed_dist = speed / dist)
summary(cars)
```

    ##      speed           dist          speed_dist    
    ##  Min.   : 4.0   Min.   :  2.00   Min.   :0.1750  
    ##  1st Qu.:12.0   1st Qu.: 26.00   1st Qu.:0.3139  
    ##  Median :15.0   Median : 36.00   Median :0.3964  
    ##  Mean   :15.4   Mean   : 42.98   Mean   :0.4769  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00   3rd Qu.:0.5208  
    ##  Max.   :25.0   Max.   :120.00   Max.   :2.0000

## Including Plots

You can also embed plots, for
example:

``` r
ggplot(cars, aes(dist, speed)) + geom_point()
```

![](ml_with_random_forest_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
