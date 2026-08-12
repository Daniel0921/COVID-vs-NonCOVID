# ============================================================
# NEURAL NETWORK DATA VISUALIZATION SUITE
# ============================================================
#
# Purpose:
# Create portfolio-ready visualizations for the COVID Year
# and Region neural-network classification project.
#
# Outputs:
#   01_COVID_Class_Distribution.png
#   02_COVID_Actual_vs_Predicted.png
#   03_COVID_Prediction_Probabilities.png
#   04_Region_Class_Distribution.png
#   05_Region_Actual_vs_Predicted.png
#   06_Region_Confusion_Matrix.png
#
# ============================================================


# ============================================================
# 1. LOAD REQUIRED LIBRARIES
# ============================================================

library(ggplot2)
library(dplyr)
library(scales)
library(readr)


# ============================================================
# 2. LOAD DATA
# ============================================================

raw_data <- read.csv(
  ""
)


covid_results <- read.csv(
  ""
)


region_results <- read.csv(
  ""
)


# ============================================================
# 3. CREATE OUTPUT DIRECTORY
# ============================================================

output_directory <- "visualizations"

dir.create(
  output_directory,
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# 4. PREPARE COVID YEAR LABELS
# ============================================================

raw_data <- raw_data %>%
  mutate(
    COVID_Label = case_when(
      COVID.Year == 1 ~ "COVID Year",
      COVID.Year == 0 ~ "Non-COVID Year",
      TRUE ~ as.character(COVID.Year)
    )
  )


covid_results <- covid_results %>%
  mutate(
    Actual_Label = case_when(
      Actual == 1 ~ "COVID Year",
      Actual == 0 ~ "Non-COVID Year",
      TRUE ~ as.character(Actual)
    ),
    
    Predicted_Label = case_when(
      Predicted == 1 ~ "COVID Year",
      Predicted == 0 ~ "Non-COVID Year",
      TRUE ~ as.character(Predicted)
    )
  )


# ============================================================
# VISUALIZATION 1
#
# COVID YEAR CLASS DISTRIBUTION
# ============================================================
#
# Question:
# How balanced is the COVID / Non-COVID target variable?
#
# ============================================================

plot_1 <- ggplot(
  raw_data,
  aes(
    x = COVID_Label,
    fill = COVID_Label
  )
) +
  
  geom_bar(
    show.legend = FALSE
  ) +
  
  geom_text(
    stat = "count",
    aes(
      label = after_stat(count)
    ),
    vjust = -0.4
  ) +
  
  labs(
    title = "COVID Year Class Distribution",
    subtitle = "Distribution of observations used for binary classification",
    x = "Class",
    y = "Number of Observations"
  ) +
  
  theme_minimal() +
  
  theme(
    plot.title = element_text(
      face = "bold"
    )
  )


print(plot_1)


ggsave(
  filename = file.path(
    output_directory,
    "01_COVID_Class_Distribution.png"
  ),
  plot = plot_1,
  width = 9,
  height = 6,
  dpi = 300
)


# ============================================================
# VISUALIZATION 2
#
# COVID YEAR ACTUAL VS PREDICTED
# ============================================================
#
# Question:
# How often did the model correctly classify COVID Year?
#
# ============================================================

covid_comparison <- covid_results %>%
  
  count(
    Actual_Label,
    Predicted_Label
  )


plot_2 <- ggplot(
  covid_comparison,
  aes(
    x = Actual_Label,
    y = n,
    fill = Predicted_Label
  )
) +
  
  geom_col(
    position = "dodge"
  ) +
  
  labs(
    title = "COVID Year: Actual vs Predicted",
    subtitle = "Neural-network classification performance on the test set",
    x = "Actual Class",
    y = "Number of Predictions",
    fill = "Predicted Class"
  ) +
  
  theme_minimal() +
  
  theme(
    legend.position = "bottom",
    plot.title = element_text(
      face = "bold"
    )
  )


print(plot_2)


ggsave(
  filename = file.path(
    output_directory,
    "02_COVID_Actual_vs_Predicted.png"
  ),
  plot = plot_2,
  width = 10,
  height = 6,
  dpi = 300
)


# ============================================================
# VISUALIZATION 3
#
# COVID PREDICTION PROBABILITIES
# ============================================================
#
# Question:
# How confident was the neural network in each prediction?
#
# The dashed line represents the 0.50 classification threshold.
#
# ============================================================

covid_probability_data <- covid_results %>%
  
  mutate(
    
    # Preserve the imported value so the script can detect
    # whether the probability was saved as a percentage string.
    probability_original = as.character(
      Probability_COVID_Year
    ),
    
    # Remove characters such as "%" and convert the result
    # into a numeric value.
    Probability_COVID_Year = readr::parse_number(
      probability_original
    ),
    
    # Convert percentage strings to the 0-1 probability scale.
    #
    # Example:
    #   "99.9998%" -> 0.999998
    #
    # Values that were already stored as decimals are unchanged.
    Probability_COVID_Year = ifelse(
      grepl(
        "%",
        probability_original
      ),
      Probability_COVID_Year / 100,
      Probability_COVID_Year
    ),
    
    # Keep all values within the valid probability range.
    Probability_COVID_Year = pmin(
      pmax(
        Probability_COVID_Year,
        0
      ),
      1
    ),
    
    Observation = seq_len(
      n()
    )
    
  )


# Confirm that the chart variable is numeric.

str(
  covid_probability_data$Probability_COVID_Year
)

summary(
  covid_probability_data$Probability_COVID_Year
)


plot_3 <- ggplot(
  covid_probability_data,
  aes(
    x = Observation,
    y = Probability_COVID_Year,
    color = Actual_Label
  )
) +
  
  geom_point(
    size = 3,
    alpha = 0.8
  ) +
  
  geom_hline(
    yintercept = 0.50,
    linetype = "dashed"
  ) +
  
  scale_y_continuous(
    labels = percent_format(
      accuracy = 1
    ),
    limits = c(
      0,
      1
    )
  ) +
  
  labs(
    title = "COVID Year Prediction Probabilities",
    subtitle = "Neural-network confidence for each test observation",
    x = "Test Observation",
    y = "Probability of COVID Year",
    color = "Actual Class",
    caption = "Dashed line represents the 50% classification threshold"
  ) +
  
  theme_minimal() +
  
  theme(
    legend.position = "bottom",
    plot.title = element_text(
      face = "bold"
    )
  )


print(plot_3)


ggsave(
  filename = file.path(
    output_directory,
    "03_COVID_Prediction_Probabilities.png"
  ),
  plot = plot_3,
  width = 11,
  height = 6,
  dpi = 300
)


# ============================================================
# VISUALIZATION 4
#
# REGION CLASS DISTRIBUTION
# ============================================================
#
# Question:
# How many observations are available for each Region?
#
# ============================================================

plot_4 <- ggplot(
  raw_data,
  aes(
    x = Region,
    fill = Region
  )
) +
  
  geom_bar(
    show.legend = FALSE
  ) +
  
  geom_text(
    stat = "count",
    aes(
      label = after_stat(count)
    ),
    vjust = -0.3,
    size = 3
  ) +
  
  labs(
    title = "Region Class Distribution",
    subtitle = "Observation count across geographic classes",
    x = "Region",
    y = "Number of Observations"
  ) +
  
  theme_minimal() +
  
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    ),
    
    plot.title = element_text(
      face = "bold"
    )
  )


print(plot_4)


ggsave(
  filename = file.path(
    output_directory,
    "04_Region_Class_Distribution.png"
  ),
  plot = plot_4,
  width = 11,
  height = 7,
  dpi = 300
)


# ============================================================
# VISUALIZATION 5
#
# REGION ACTUAL VS PREDICTED
# ============================================================
#
# Question:
# Which regions were classified correctly or incorrectly?
#
# ============================================================

region_plot_data <- region_results %>%
  
  mutate(
    Observation = seq_len(
      n()
    )
  ) %>%
  
  select(
    Observation,
    Actual_Region,
    Predicted_Region
  )


region_long <- region_plot_data %>%
  
  tidyr::pivot_longer(
    cols = c(
      Actual_Region,
      Predicted_Region
    ),
    names_to = "Result_Type",
    values_to = "Region"
  ) %>%
  
  mutate(
    Result_Type = recode(
      Result_Type,
      Actual_Region = "Actual",
      Predicted_Region = "Predicted"
    )
  )


plot_5 <- ggplot(
  region_long,
  aes(
    x = Observation,
    y = Region,
    shape = Result_Type
  )
) +
  
  geom_point(
    size = 3,
    alpha = 0.8
  ) +
  
  labs(
    title = "Region Classification: Actual vs Predicted",
    subtitle = "Comparison of true and predicted geographic classes",
    x = "Test Observation",
    y = "Region",
    shape = "Result"
  ) +
  
  theme_minimal() +
  
  theme(
    legend.position = "bottom",
    plot.title = element_text(
      face = "bold"
    )
  )


print(plot_5)


ggsave(
  filename = file.path(
    output_directory,
    "05_Region_Actual_vs_Predicted.png"
  ),
  plot = plot_5,
  width = 12,
  height = 7,
  dpi = 300
)


# ============================================================
# VISUALIZATION 6
#
# REGION CONFUSION MATRIX HEATMAP
# ============================================================
#
# Question:
# Which Region classes were confused with one another?
#
# ============================================================

region_confusion <- table(
  Predicted = region_results$Predicted_Region,
  Actual = region_results$Actual_Region
)


region_confusion_df <- as.data.frame(
  region_confusion
)


names(
  region_confusion_df
) <- c(
  "Predicted",
  "Actual",
  "Count"
)


plot_6 <- ggplot(
  region_confusion_df,
  aes(
    x = Actual,
    y = Predicted,
    fill = Count
  )
) +
  
  geom_tile(
    color = "white"
  ) +
  
  geom_text(
    aes(
      label = Count
    ),
    size = 4
  ) +
  
  scale_fill_gradient(
    low = "white",
    high = "steelblue"
  ) +
  
  labs(
    title = "Region Neural Network Confusion Matrix",
    subtitle = "Actual region compared with predicted region",
    x = "Actual Region",
    y = "Predicted Region",
    fill = "Count"
  ) +
  
  theme_minimal() +
  
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    ),
    
    panel.grid = element_blank(),
    
    plot.title = element_text(
      face = "bold"
    )
  )


print(plot_6)


ggsave(
  filename = file.path(
    output_directory,
    "06_Region_Confusion_Matrix.png"
  ),
  plot = plot_6,
  width = 11,
  height = 8,
  dpi = 300
)


# ============================================================
# 5. COMPLETION MESSAGE
# ============================================================

cat(
  "\n",
  "============================================================\n",
  "NEURAL NETWORK VISUALIZATION SUITE COMPLETE\n",
  "============================================================\n",
  "\n",
  "6 portfolio-ready visualizations created.\n",
  "\n",
  "Output directory:\n",
  output_directory,
  "\n\n",
  "Files Created:\n",
  "01_COVID_Class_Distribution.png\n",
  "02_COVID_Actual_vs_Predicted.png\n",
  "03_COVID_Prediction_Probabilities.png\n",
  "04_Region_Class_Distribution.png\n",
  "05_Region_Actual_vs_Predicted.png\n",
  "06_Region_Confusion_Matrix.png\n",
  "\n",
  "============================================================\n"
)
