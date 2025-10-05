rm(list = ls())
library(readxl)
library("survival")
library("survminer")
library(dplyr)
art_data <- read_excel("~/Downloads/max_discount_data.xlsx")

art_data <- art_data %>%
  mutate(Painting_ID = paste(`Artist Name`, Image_No, `Listed Date`, sep = "_"))
max_days<- max(art_data$Days_Since_Listing, na.rm = TRUE)
max_days
epsilon <- 0.01
art_data_tv <- art_data %>%
  arrange(Painting_ID, Days_Since_Listing) %>%
  group_by(Painting_ID) %>%
  mutate(
    start    = Days_Since_Listing,
    stop     = lead(Days_Since_Listing),
    # ensure last row has positive length
    stop     = if_else(is.na(stop) | stop <= start, start + epsilon, stop),
    # event only on the last row of each painting
    sold_any = as.integer(any(Sold == 1, na.rm = TRUE)),
    event    = if_else(row_number() == n(), sold_any, 0L)
  ) %>%
  ungroup() %>%
  select(-sold_any)



res.cox <- coxph(Surv(start, stop, event) ~
                   Actual_Price + max_discount_by_week + Rating + Review +
                   Is_Rare_Find + Admirers + Actual_Width + Actual_Height +
                   Canvas + Mixed_Media + Oil + Acrylic + Framed,
                 data = art_data_tv)
summary(res.cox)

# sf <- survfit(res.cox, newdata = art_data_tv)
# summary(sf)$table[, "median"]
covar<- c("Actual_Price", "max_discount_by_week", "Rating", "Review",
          "Is_Rare_Find", "Admirers", "Actual_Width", "Actual_Height",
          "Canvas", "Mixed_Media", "Oil", "Acrylic", "Framed")

# art_data_first <- art_data_tv %>%
#   group_by(Painting_ID) %>%
#   slice_min(order_by = start) %>%
#   ungroup()

art_data_median <- art_data_tv %>%
  group_by(Painting_ID) %>%
  summarise(
    across(covar, ~ median(.x, na.rm = TRUE)),
    .groups = "drop"
  )
library(ggplot2)
ggplot(art_data_median, aes(x = Actual_Price)) +
  geom_histogram(bins = 40, color = "white") +
  scale_x_continuous(labels = scales::label_dollar()) +
  labs(title = "Price Distribution (one row per painting)",
       x = "Price", y = "Count")
ggplot(art_data_median, aes(x = Rating)) +
  geom_histogram(bins = 1, color = "white") +
  labs(title = "Rating Distribution (one row per painting)",
       x = "Rating", y = "Count")
#min 0, max 204
art_data_median<- art_data_median %>%
  mutate(
    Has_Discount = ifelse(max_discount_by_week > 0, "With Discount", "No Discount"),
    Has_Discount = factor(
      Has_Discount,
      levels = c("With Discount", "No Discount"),  # correct order
      ordered = TRUE
    )
  )
price_cut  <- 245
rating_cut <- 4.5
art_data_median <- art_data_median %>%
  mutate(
    Price_Rating_Class = case_when(
      Actual_Price > price_cut  & Rating >  rating_cut ~ "High price · High rating",
      Actual_Price > price_cut  & Rating <= rating_cut ~ "High price · Low rating",
      Actual_Price <= price_cut  & Rating >  rating_cut ~ "Low price · High rating",
      TRUE                                   ~ "Low price · Low rating"
    ),
    # (optional) make it an ordered factor for nice plotting/tables
    Price_Rating_Class = factor(Price_Rating_Class, levels = c(
      "High price · High rating",
      "High price · Low rating",
      "Low price · High rating",
      "Low price · Low rating"
    ))
  )
# art_data_median <- art_data_median %>%
#   mutate(
#     Price_Class = case_when(
#       Actual_Price <= 50 ~ "<=$50",
#       Actual_Price > 50 & Actual_Price <= 190 ~ "$51-$190",
#       Actual_Price > 190 & Actual_Price <= 310 ~ "$191-$310",
#       Actual_Price > 310 ~ ">$310",
#       TRUE ~ NA_character_
#     ),
#     Price_Class = factor(
#       Price_Class,
#       levels = c("<=$50", "$51-$190", "$191-$310", ">$310"),
#       ordered = TRUE
#     )
#   )
art_data_median <- art_data_median %>%
  mutate(
    Rating_Class = case_when(
      Rating <= 4.5 ~ "Low Rating",
      Rating > 4.5 ~ "High Rating"
    ),
    Rating_Class = factor(
      Rating_Class,
      levels = c("Low Rating", "High Rating"),
      ordered = TRUE
    )
  )
median_price<- median(art_data_median$Actual_Price, na.rm = TRUE)
median_price
newdata_all <- art_data_median%>%
  summarise(across(c(Actual_Price, max_discount_by_week, Rating, Review,
                     Is_Rare_Find, Admirers, Actual_Width, Actual_Height,
                     Canvas, Mixed_Media, Oil, Acrylic, Framed),
                   median, na.rm=TRUE))
newdata_all$Group <- "All Paintings"
sf_all <- survfit(res.cox,newdata = newdata_all)

newdata_discount <- art_data_median%>%
  group_by(Has_Discount) %>%
  summarise(across(c(Actual_Price, max_discount_by_week, Rating, Review,
                     Is_Rare_Find, Admirers, Actual_Width, Actual_Height,
                     Canvas, Mixed_Media, Oil, Acrylic, Framed),
                   median, na.rm = TRUE)) %>%
  ungroup()%>%
  rename(Group = Has_Discount)

sf_discount <- survfit(res.cox, newdata = newdata_discount)

newdata_rating <- art_data_median %>%
  group_by(Rating_Class) %>%
  summarise(across(c(Actual_Price, max_discount_by_week, Rating, Review,
                     Is_Rare_Find, Admirers, Actual_Width, Actual_Height,
                     Canvas, Mixed_Media, Oil, Acrylic, Framed),
                   median, na.rm = TRUE)) %>%
  ungroup()%>%
  rename(Group = Rating_Class)

sf_rating <- survfit(res.cox, newdata = newdata_rating)
newdata_price_rating <- art_data_median %>%
  group_by(Price_Rating_Class) %>%
  summarise(across(c(Actual_Price, max_discount_by_week, Rating, Review,
                     Is_Rare_Find, Admirers, Actual_Width, Actual_Height,
                     Canvas, Mixed_Media, Oil, Acrylic, Framed),
                   median, na.rm = TRUE)) %>%
  ungroup()%>%
  rename(Group = Price_Rating_Class)
sf_price_rating <- survfit(res.cox, newdata = newdata_price_rating)
# Survival probabilities at time points
summary(sf_all, times = c(30,60,90,120,150,180,210))
summary(sf_discount, times = c(30,60,90,120,150,180,210))
summary(sf_rating, times = c(30,60,90,120,150,180,210))
summary(sf_price_rating, times = c(30,60,90,120,150,180,210))
# Overall median
summary(sf_all)$table["median"]

# By discount vs no-discount
summary(sf_discount)$table[, "median"]

# By price level
summary(sf_rating)$table[, "median"]

# By price level
summary(sf_price_rating)$table[, "median"]
# Overall
# ggsurvplot(sf_all, conf.int = TRUE, legend.labs = "All Paintings",
#            ggtheme = theme_minimal())

# Discount vs No Discount
#For overall average painting
ggsurvplot(sf_all, data = art_data_tv,
           conf.int = FALSE,
           legend.title = "All",
           legend.labs = "All Paintings",
           ggtheme = theme_minimal(),
           break.time.by = 30,
           xlab = "Days",
           font.legend = c(14, "plain", "black"),  # size, style, color
           font.x = c(14, "plain", "black"),       # x-axis label
           font.y = c(14, "plain", "black"),       # y-axis label
           font.tickslab = c(12, "plain", "black") # axis ticks
)
# tick marks every 30 days)

# 
# # Discount vs No Discount
# ggsurvplot(sf_discount, data = art_data_tv,
#            conf.int = TRUE,
#            legend.title = "Discount",
#            legend.labs = newdata_discount$Group,
#            ggtheme = theme_minimal(),
#            xlim = c(0, 210),         # set x-axis range
#            break.time.by = 30 )       # tick marks every 30 days)
ggsurvplot(sf_discount, data = art_data_tv,
           conf.int = FALSE,
           legend.title = "Discount Class",
           legend.labs = newdata_discount$Group,
           ggtheme = theme_minimal(),
           xlim = c(0, 210),         # set x-axis range
           break.time.by = 30,
           xlab = "Days",# tick marks every 30 days
           # extra font controls
           font.legend = c(14, "plain", "black"),  # size, style, color
           font.x = c(14, "plain", "black"),       # x-axis label
           font.y = c(14, "plain", "black"),       # y-axis label
           font.tickslab = c(12, "plain", "black") # axis ticks
)
ggsurvplot_combine(
  list(sf_all, sf_discount),
  data = art_data_tv,
  conf.int = FALSE,
  legend.title = "Groups",
  legend.labs = c("All Paintings",
                  paste(newdata_discount$Group)),
  ggtheme = theme_bw(),   # repeat black for all 3 curves
  linetype = "strata",         # each group gets its own line type
  xlim = c(0, 210),
  break.time.by = 30,
  xlab = "Days",
  # extra font controls
  font.legend = c(14, "plain", "black"),  # size, style, color
  font.x = c(14, "plain", "black"),       # x-axis label
  font.y = c(14, "plain", "black"),       # y-axis label
  font.tickslab = c(12, "plain", "black") # axis ticks
)
#rating class
ggsurvplot(sf_rating, data = art_data_tv,
           conf.int = FALSE,
           legend.title = "Rating Class",
           legend.labs = newdata_rating$Group,
           ggtheme = theme_minimal(),
           xlim = c(0, 210),         # set x-axis range
           break.time.by = 30,
           xlab = "Days",# tick marks every 30 days
           # extra font controls
           font.legend = c(14, "plain", "black"),  # size, style, color
           font.x = c(14, "plain", "black"),       # x-axis label
           font.y = c(14, "plain", "black"),       # y-axis label
           font.tickslab = c(12, "plain", "black") # axis ticks
)


#price_rating Class
ggsurvplot(sf_price_rating, data = art_data_tv,
           conf.int = FALSE,
           legend.title = "Price_Rating Class",
           legend.labs = newdata_price_rating$Group,
           ggtheme = theme_minimal(),
           xlim = c(0, 210),         # set x-axis range
           break.time.by = 30,
           xlab = "Days",# tick marks every 30 days
           # extra font controls
           font.legend = c(14, "plain", "black"),  # size, style, color
           font.x = c(14, "plain", "black"),       # x-axis label
           font.y = c(14, "plain", "black"),       # y-axis label
           font.tickslab = c(12, "plain", "black") # axis ticks
           )
