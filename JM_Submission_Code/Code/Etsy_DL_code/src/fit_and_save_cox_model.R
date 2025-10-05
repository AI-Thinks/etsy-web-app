rm(list = ls())
library(readxl)
library("survival")
library(dplyr)
library(jsonlite)

# Read data
art_data <- read_excel("/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/JM_Submission_Code/Data/max_discount_data.xlsx")

# Prepare data for Cox model
art_data <- art_data %>%
  mutate(Painting_ID = paste(`Artist Name`, Image_No, `Listed Date`, sep = "_"))

max_days <- max(art_data$Days_Since_Listing, na.rm = TRUE)
epsilon <- 0.01

art_data_tv <- art_data %>%
  arrange(Painting_ID, Days_Since_Listing) %>%
  group_by(Painting_ID) %>%
  mutate(
    start    = Days_Since_Listing,
    stop     = lead(Days_Since_Listing),
    stop     = if_else(is.na(stop) | stop <= start, start + epsilon, stop),
    sold_any = as.integer(any(Sold == 1, na.rm = TRUE)),
    event    = if_else(row_number() == n(), sold_any, 0L)
  ) %>%
  ungroup() %>%
  select(-sold_any)

# Fit Cox model
res.cox <- coxph(Surv(start, stop, event) ~
                   Actual_Price + max_discount_by_week + Rating + Review +
                   Is_Rare_Find + Admirers + Actual_Width + Actual_Height +
                   Canvas + Mixed_Media + Oil + Acrylic + Framed,
                 data = art_data_tv,
                 model = TRUE)

print(summary(res.cox))

# Define covariates
covar <- c("Actual_Price", "max_discount_by_week", "Rating", "Review",
          "Is_Rare_Find", "Admirers", "Actual_Width", "Actual_Height",
          "Canvas", "Mixed_Media", "Oil", "Acrylic", "Framed")

# Compute median values per painting, then overall medians
art_data_median <- art_data_tv %>%
  group_by(Painting_ID) %>%
  summarise(
    across(all_of(covar), ~ median(.x, na.rm = TRUE)),
    .groups = "drop"
  )

# Compute overall median values across all paintings
median_values <- art_data_median %>%
  summarise(across(all_of(covar), ~ median(.x, na.rm = TRUE)))

print("Median values for covariates:")
print(median_values)

# Save the fitted Cox model
output_dir <- "/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/JM_Submission_Code/Code/Etsy_DL_code/src"

# Remove the data environment reference to make model self-contained
res.cox$call$data <- NULL

saveRDS(res.cox, file = file.path(output_dir, "res_cox_fitted.rds"))
print(paste("Saved Cox model to:", file.path(output_dir, "res_cox_fitted.rds")))

# Save median values as JSON for easy access in Python
median_list <- as.list(median_values)
write_json(median_list, file.path(output_dir, "cox_median_values.json"), auto_unbox = TRUE, pretty = TRUE)
print(paste("Saved median values to:", file.path(output_dir, "cox_median_values.json")))

print("Model fitting and saving completed successfully!")

