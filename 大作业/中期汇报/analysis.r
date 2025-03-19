library(psych)
library(ShinyItemAnalysis)

# Read preExpStd.csv file
data_subset <- read.csv("preExpStd.csv", header = TRUE, sep = ",")


# difficulty and discrimination plot
DDplot(data_subset, discrim = "ULI", k = 3, l = 1, u = 3)

# Cronbach alpha
psych::alpha(data_subset)

# traditional item analysis table
ItemAnalysis(data_subset)

# Run the interactive Shiny app without background job
startShinyItemAnalysis(background = FALSE)