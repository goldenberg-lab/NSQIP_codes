# 02_eda.
# Delvin So
library(tidyverse)
library(naniar)

# setting up directories
setwd("~/Documents/OneDrive - SickKids/nsqip/")
figure_dir <- 'output/figures/'
source("analysis/nsqip_helpers.R")

# reading in the data
dat_all_raw <- read.csv("data/output/combined_raw.csv", stringsAsFactors = FALSE) %>% as_tibble() # the data 
yr_vars_all <- read_csv("output/var_names_yrs.csv")                 # variable names
master_key <- read_csv("output/master_key.csv")                     # master key linking variable names to descriptions

# ---- shared variables ----
# how many variables are shared across the 7 years?

png(filename = paste0(figure_dir, "shared_variables.png"), width = 640, height = 480)
yr_vars_all %>% 
  count(vars, sort = TRUE) %>% count(n) %>%
  ggplot(aes(x = n, y = nn)) +
  geom_col(colour = "black", alpha = 0.9) +
  geom_label(aes(label = nn)) +
  scale_x_continuous(breaks = seq(1, 7)) + 
  ggtitle("Counts of Shared Variables Across 2012 - 2018", 'NSQIP-P 2012 - 2017') +
  coord_flip() +
  labs(x = 'Number of Years', y = 'Number of Shared Variables')
dev.off()


# ---- cleaning ----

# replace all nulls, empty space and -99 with NAs
dat_all <- dat_all_raw %>% 
  # select(cva, renafail) %>% # for testing
  mutate_all(~ str_to_lower(.)) %>%
  mutate_all(~ replace(., . %in% c("", "null", "-99"), NA))

# save down (uncomment below if you want to re-process the data)
# write_csv(dat_all, "data/output/combined_raw_clean.csv")

# --- read in the data ----
dat_all <- read.csv("data/output/combined_raw_clean.csv", stringsAsFactors = FALSE)
dat_all <- as_tibble(dat_all)

# assume no complication if missing
dat_all$sdehis <- ifelse(is.na(dat_all$sdehis), "no complication", dat_all$sdehis)  
# ----- eda -----
# how many observations by year?
dat_all %>% count(operyr)

# operyr      n
# <int>  <int>
#   1   2012  51008
# 2   2013  63387
# 3   2014  68838
# 4   2015  84056
# 5   2016 101887
# 6   2017 113922
# 7   2018 119486

png(filename = paste0(figure_dir, "obs_by_yr.png"), width = 640, height = 480)
dat_all %>% 
  count(operyr) %>%
  ggplot(aes(x = operyr, y = n)) + 
  geom_col() + 
  coord_flip() + 
  geom_label(aes(label = n)) +
  scale_x_continuous(breaks = seq(2012, 2018, by = 1)) +
  labs(x = "Year", y = "# of Observations") +
  ggtitle("Observations by Year", "NSQIP-P 2012 - 2018")
dev.off()

# ---- get a feel for missingnesss ----

# by year
miss_summ_by_yr <- dat_all %>% 
  group_by(operyr) %>%
  miss_var_summary() 
  
# overall
miss_summ_overall <- dat_all %>% 
  miss_var_summary() 

# which variables have ANY data that's missing?
miss_summ_overall %>% filter(n_miss > 0 ) %>% print(n = 400)

# which variables do not have any missing data?
miss_summ_overall %>% filter(n_miss == 0 ) %>% print(n = 100)

# 81 variables have zero missing data (need to sanity check whether this is true or not)
vars_no_miss <- miss_summ_overall %>% filter(n_miss == 0 ) %>% pull(variable)
length(vars_no_miss) # 81

# which variables have low missing data (let's say 5% threshold)
vars_low_miss <- miss_summ_overall %>% filter(pct_miss < 5 )
nrow(vars_low_miss) # 101

# descr. statistics of missingness of each of the presumed core variables across the 7 years
low_miss_descr_stats <- miss_summ_by_yr %>% group_by(variable) %>% 
  summarize(max_miss = max(pct_miss),
            min_miss = min(pct_miss),
            stdev_miss = sqrt(var(pct_miss)),
            n = n()) %>% 
  arrange(-max_miss) %>% 
  filter(variable %in% vars_low_miss$variable) %>% 
  left_join(master_key, by = c("variable" = "vars")) %>%
  print(n = 400) 

low_miss_descr_stats
# write_csv(low_miss_descr_stats, "output/low_missingness_variables_descriptive_stats.csv")

# 103 variables, year is missing. 
# which(!vars_low_miss$variable %in% miss_summ_by_yr$variable) # 26
# vars_low_miss$variable[which(!vars_low_miss$variable %in% miss_summ_by_yr$variable)]

# distribution of missing variables, overall 
png(filename = paste0(figure_dir, "missing_distr_overall.png"), width = 640, height = 480)
miss_summ_overall %>% 
  ggplot(aes(x = pct_miss)) + 
  geom_histogram(aes(fill = ..count..), alpha = 0.9) +
  scale_fill_viridis_c() + 
  ggtitle("Distribution of Missing Data", "NSQIP-P 2012 - 2018") + 
  labs(x = "Percentage Missing", y = "Counts") +
  guides(fill = FALSE)
dev.off()

# distribution of missing variables, by year
png(filename = paste0(figure_dir, "missing_distr_by_yr.png"), width = 640, height = 480)
miss_summ_by_yr  %>%
  ggplot(aes(x = pct_miss)) + 
  geom_histogram(aes(fill = ..count..), alpha = 0.9) +
  scale_fill_viridis_c() + 
  ggtitle("Distribution of Missing Data", "NSQIP-P 2012 - 2018") + 
  labs(x = "Percentage Missing", y = "Counts") + 
  facet_wrap(operyr ~ ., scales = "free") + 
  guides(fill = FALSE)
dev.off()

# ----- cpt codes (surgical procedure billing code) -----

dat_cpt <- dat_all %>% select(operyr, contains("cpt"))

names(dat_cpt)
length(names(dat_cpt)) # 26 variables with 'cpt' 

# number of unique cpt codes
dat_cpt %>% 
  distinct(cpt) %>%
  count() 
# 716

# distribution of cpt codes
dat_cpt %>% count(cpt) %>%
  ggplot(aes(x = log10(n))) + 
  geom_histogram(bins = 10) 

# missingness across the 26 CPT variables
png(filename = paste0(figure_dir, 'cpt_vars_missingness.png'), height = 480, width = 640)
dat_cpt %>%
  miss_var_summary() %>% 
  arrange(variable) %>%
  print(n = 25) %>%
  ggplot(aes(x = pct_miss)) +
  geom_histogram() +
  ggtitle("Distribution of the Percentage of Missing CPT Code Variables", "NSQIP-P 2012-2018") +
  labs(x = "% Missingness", y = "")
dev.off()


# prncptx code counts
dat_all %>% count(prncptx, sort = TRUE)

# do all prncptx descriptions account for all cpt codes?
# ie. are prncpt descriptions unique to cpt codes?
# the counts of prncptx by cpt should be identical to the counts of cpt if so..

prncptx_joined_counts <- dat_all %>%
  count(prncptx, cpt, sort = TRUE) %>% 
  na.omit()  %>% 
  rename(joint_counts = n) %>%
  left_join(dat_all %>% 
              count(prncptx, sort = TRUE) %>% 
              rename(prncptx_counts = n),
            by = c("prncptx")) %>%
  mutate(is_equal = joint_counts == prncptx_counts, total_counts = sum(joint_counts))

prncptx_joined_counts
# sum(prncptx_joined_counts$is_equal)/nrow(prncptx_joined_counts)


# ----cumulative frequencies of cpt codes ----
cumulative_cpt_freqs <- dat_all %>% 
  count(cpt) %>%
  arrange(-n) %>%
  mutate(cumulative_sum = cumsum(n),
         row_num = row_number(),
         grand_total = max(cumulative_sum),
         perc = cumulative_sum/grand_total * 100,
         perc_of_total_vars = row_num/max(row_num) * 100) 

# does the pareto principle apply here?
cumulative_cpt_freqs   %>% 
  filter(perc_of_total_vars <= 20.0) %>% 
  # arrange(-perc_of_total_vars)
  print(n = 200)
# ye
# 143 30462   966         492364     143      602584  81.7             20.0 

png(filename = paste0(figure_dir, "pareto_all_cpt_codes.png"), width = 640, height = 480)
cumulative_cpt_freqs %>%
  ggplot(aes(x = row_num, y = perc)) +
  # geom_hline(yintercept = c(25, 50, 80, 99), alpha = 0.8, linetype = "dashed") +
  # geom_hline(aes(yintercept = c(25, 50))) + 
  geom_line() +
  labs(y = "Cumulative Frequency (%)", x = "# of CPT Codes (In descending order)") +
  ggtitle("Cumulative Frequency of CPT Codes (5 digits)", subtitle = "NSQIP-P 2012 - 2018")
dev.off()

# ---- mapping cpt codes to surgery ----

# surgery codes
code_start <- c(10000, 10040, 20000, 30000, 33010, 38100, 39000, 40490, 50010, 54000,
                55920, 56405, 59000, 60000, 61000, 65091, 69000)
code_end <- c(10022, 19499, 29999, 32999, 37799, 38999, 39599, 49999, 53899, 55899, 55980, 58999,
              59899, 60699, 64999, 68899, 69979)
categories <- c("general", "integumentary", "musculoskeletal", "respiratory", "cardiovascular", "hemic_lymphatic", "mediastinum_diaphragm",
                "digestive", "urinary", "male_genital", "reproductive_intersex", "female_genital", "maternity_care_delivery", "endocrine", "nervous",
                "eye_and_ocular", "auditory")

surgery_cpt <- tibble(code_start, code_end, categories)
surgery_cpt


# ---- more exploration of low missingness variables ---- 

vars_low_miss %>% 
  left_join(master_key, by = c("variable" = "vars")) %>% 
  print(n = 100) %>%
  write_csv("output/low_missingness_vars_with_descr.csv")


dat_core <- dat_all[c("caseid", vars_low_miss$variable)]

# write_csv(dat_core, "output/dat_core.csv") # the low missingness variables (across years!!)

dat_all %>% select(get_def("superficial")$vars) %>% .[,1:3] %>%
  ggplot(aes(x = dsupinfec)) + 
  geom_histogram()

dat_all %>% select(get_def("superficial")$vars)  %>%
  count(supinfec, dsupinfec) %>% 
  print(n = 500)


