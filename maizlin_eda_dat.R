# ----- Exploratory Analyses using Maizlin et al. 2017 variables and outcomes -----
# - Creates various figures (in ./figures/) for exploring missingness and outcomes used in their paper,
#    by year for 2012 to 2018 (original paper looks at 2012 - 2014)
# - Creates X matrix (output/maizlin_x.csv) and and 5 outcomes (output/maizlin_y.csv) used in their paper
# - Requires 02_dirty_eda.R to be run first to obtain 'output/combined_raw_clean.csv'

library(tidyverse)
library(sqldf) 
library(naniar)

# ---- setting up ----
setwd("~/Documents/OneDrive - SickKids/nsqip/")
figure_dir <- 'output/figures/'
source("analysis/nsqip_helpers.R")

# ---- reading in the data ----
dat_all <- read.csv("output/combined_raw_clean.csv",
                    stringsAsFactors = FALSE) %>% as_tibble()       # the data 
yr_vars_all <- read_csv("output/yr_vars.csv")                       # variable names
master_key <- read_csv("output/master_key.csv")                     # master key linking variable names to descriptions
 

# the outcomes 
maizlin_outcomes <-  c("supinfec", "wndinfd", "orgspcssi", "sdehis", "dehis")


# --- maizlin et al. 2017  -----
# ---- outcomes -----


# ---- overall outcomes ----
maizlin_outcomes_cts <- dat_all %>% 
  select(caseid, operyr, maizlin_outcomes) %>%
  gather(key = outcome, value = variable, 
         -one_of("caseid", "operyr"))  %>%
  count(outcome, variable) %>%
  group_by(outcome) %>%
  mutate(perc= prop.table(n) * 100) %>%
  mutate(total_n = sum(n)) %>%
  left_join(master_key, by = c( "outcome" = "vars")) %>%
  mutate(variable_label = str_replace_all(variable_label, "occurrences ", ""),
         variable_label = str_replace_all(variable_label, "\\\r", " "))

maizlin_outcomes_cts

png(filename = paste0(figure_dir, 'wnd_class_freqs_overall.png'), height = 640, width = 860)
maizlin_outcomes_cts %>%
  ggplot(aes(x = variable, y = perc, group = variable)) +
  geom_col(aes(fill = outcome), colour = "black", position = "dodge") +
  geom_label(aes(label = paste0(round(perc, 2), "%")),
             position = position_dodge(width = 1), hjust = -0.01) +
  scale_x_discrete(labels = scales::wrap_format(10)) +
  coord_flip() +
  guides(colour = FALSE) +
  ggtitle("Overall Wound Class Frequencies", "NSQIP-P 2012 - 2018") +
  labs(x = "", y = "Percentage (%)")  +
  scale_fill_brewer(palette = "Dark2") +
  theme(legend.direction = "horizontal", legend.position = "top") +
  scale_y_continuous(limits = c(0, 110), breaks = seq(0, 100, by = 10)) +
  facet_wrap(~ variable_label, scales = "free", ncol = 2) +
  guides(fill = FALSE) 
dev.off()


# Maizlin et al. 2018 reports 1.0% sSSIs and 0.8% oSSI for 2012 - 2014. Has not changed 

# ----Cross-tab of outcomes with traditional ASA wound classification (wndclas) ----
maizlin_outcomes %>%
  map(function(outcome){
    print(outcome)
    outcome <- ensym(outcome)
    dat_all %>% 
      count(!!outcome, wndclas) %>%
      mutate(perc= prop.table(n) * 100) %>%
      # rename(variable = !!outcome) %>%
      select(-perc) %>%
      spread(wndclas, n)
  })

# ---- outcomes by year ----

maizlin_out_mdl <- dat_all %>% 
  select(caseid, operyr, maizlin_outcomes) %>%
  mutate_at(vars(-one_of("operyr", "caseid")), ~  case_when(
    . == "no complication" ~ 0, # if no complication, negative class
    is.na(.) ~ NA_real_,       # if NA, then leave as NA
    TRUE ~ 1                   # finally, if none of the above then it must be a positive class
  )) 

# sanity check 
maizln_class_cts_overall <- maizlin_out_mdl[, -c(1:2)] %>% 
  apply(., 2, table)

# positive class counts by year
maizlin_pos_class_cts_yr <- maizlin_out_mdl %>%
  gather(key = outcome, value = variable, 
         -one_of("caseid", "operyr")) %>%
  group_by(operyr) %>%
  count(outcome, variable) %>%
  spread(variable, n) %>% 
  mutate(perc_positive = (`1`/`0`) * 100)  %>%
  print(n = 40)

# 2012 sdehis and 2018 have no positive outcomes.

maizlin_pos_class_cts_yr

png(filename = paste0(figure_dir, "maizlin_pos_class_cts_yr.png"), width = 860, height = 640)
maizlin_pos_class_cts_yr  %>% 
  ggplot(aes(x = operyr, y = perc_positive)) +
  geom_col(aes(fill = perc_positive)) +
  geom_label(aes(label = round(perc_positive, 3)), size = 3) + 
  facet_wrap(~ outcome, scales = "free") +
  scale_y_continuous(limits = c(0, 1.5)) +
  scale_x_continuous(breaks = seq(2012, 2018, by = 1)) + 
  labs(x = "Operating Year", y = "Percentage of Positive Class") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Maizlin et al. Outcomes - Positive Class %s by Year",
          "NSQIP-P 2012 - 2018") +
  guides(fill = FALSE)
# coord_flip()
dev.off()

# ---- combinations of SSI outcomes for multi task learning ----

dat_all %>%
  select(maizlin_outcomes) %>%
  count(supinfec, wndinfd, orgspcssi, sdehis, dehis) %>%
  mutate(total = nrow(dat_all)) %>% 
  mutate(perc = n/total * 100) %>%
  arrange(-perc)

#  -----positive group breakdown in Maizlin by general cpt codes ----


# higher level, surgical coding for cpt - what is this
dat_all$cpt_gen <-str_sub(dat_all$cpt, 1, 3)

# cpt_by_pos_class
cpt_by_pos_class <- dat_all %>%
  select(caseid, operyr, maizlin_outcomes, cpt) %>%
  gather(key = outcome, value = variable, supinfec:dehis) %>%
  count(variable, cpt, sort = TRUE) %>% # count by cpt, by variable and outcome is the same
  filter(variable != "no complication") %>%
  group_by(variable) %>%
  mutate(perc= prop.table(n) * 100) %>%
  # get total counts of the class
  mutate(total_n = sum(n)) #%>%

cpt_by_pos_class %>% arrange(-total_n)# %>% distinct(variable)
cpt_by_pos_class2 %>% arrange(-total_n)# %>% distinct(outcome)

identical(cpt_by_pos_class %>% arrange(-total_n),
          cpt_by_pos_class2 %>% arrange(-total_n))


cpt_by_pos_class_prto <- cpt_by_pos_class%>% 
  group_by(variable) %>%
  mutate(roll_perc = cumsum(perc),
         row_num = row_number(),
         perc_of_total_vars = row_num/max(row_num) * 100)

# how much does 20% of the top 20 CPT codes cover?

cpt_by_pos_class_prto %>%
  filter(perc_of_total_vars <= 20.999 & perc_of_total_vars >= 19.000) %>%
  filter(perc_of_total_vars == max(perc_of_total_vars)) 

# 20% of CPT codes capture between 67.5 and 83.8% of the overall # of CPT codes

png(filename = paste0(figure_dir, "pareto_positive_ssi_cts_by_all_cpt.png"), width = 640, height = 480)
cpt_by_pos_class_prto %>%
  ggplot(aes(x = row_num, y = roll_perc)) +
  geom_line() + 
  facet_wrap(~ variable) +
  labs(x = "Number of CPT Codes", y = "Cumulative Frequency") +
  ggtitle("Pareto Plots of CPT Code Distribution by Outcome",
          "NSQIP-P 2012 - 2018")
dev.off()

png(filename = paste0(figure_dir, "positive_ssi_cts_by_all_cpt.png"), width = 640, height = 480)
# png(filename = paste0(figure_dir, "positive_ssi_cts_by_surgical_cpt.png"), width = 640, height = 480)
cpt_by_pos_class %>%
  group_by(variable) %>%
  mutate(n = log(n)) %>%
  ggplot(aes(x = n)) + 
  geom_histogram() + 
  facet_wrap(~ variable, scales = "free", nrow = 3) +
  # ggtitle("Distribution of Positive Class Counts by Overall CPT Codes")
  ggtitle("Distribution of Positive Class Counts by All CPT Codes")
dev.off()
# matches up with Erik's ~150 general cpt codes? or was it 160.

dat_all %>% distinct(prncptx, cpt) %>% filter(str_detect(cpt, "449"))


# ---- outcome labels for modeling ----

mdl_maizlin <- dat_all %>% 
  select(caseid, operyr, maizlin_outcomes) %>% 
  mutate_at(vars(-one_of("operyr", "caseid")), ~  case_when(
    . == "no complication" ~ 0, # if no complication, negative class
    is.na(.) ~ NA_real_,              # if NA, then leave as NA
    TRUE ~ 1                   # finally, if none of the above then it must be a positive class
  )) 

# sanity check for positive class counts
apply(dat_all %>% 
        select(maizlin_outcomes),
      2,
      table)

# supinfec wndinfd orgspcssi sdehis  dehis
# [1,]   596357    1257    597484 597495 600464
# [2,]     6227  601327      5100   5089   2120
apply(mdl_maizlin[, -c(1:2)], 2, table)
# supinfec wndinfd orgspcssi sdehis  dehis
# 0   596357  601327    597484 597495 600464
# 1     6227    1257      5100   5089   2120

# sanity check to make sure the positive classes match up
mdl_maizlin %>% filter(wndinfd == 1) %>% 
  left_join(dat_all %>%
              select(caseid, maizlin_outcomes) %>% 
              filter(wndinfd == "deep incisional ssi"), 
            by = c("caseid")) %>% 
  print(n = 500)



# ---- extracting maizlin variables using my data dictionary ----

# TODO: input a vector and collapse the variables into a single regex term?
maizlin_vars <- get_def("^cpt$|diabetes|premature|ventilator dependence|occurrences pneumonia|\n
                        |oxygen support|tracheostomy|esophageal|biliary|cardiac risk|occurrences acute renal fail|currently on dialysis|\n
                        |intracranial hemorrhage|developmental delay|cerebral palsy|immune disease|steroid use|bone marrow transplant|\n
                        |solid organ|open wound|weight loss|nutritional support|bleeding disorders|hematologic disorder|chemotherapy for malignancy|\n
                        |radiotherapy for malignancy|shock within 48|inotropic|previous cpr|prior operation|congenital malformation|blood transfusions|\n
                        |childhood malignancy|case status|asa classification|serum albumin|pre-operative wbc|duration from anesthesia start|duration from surgery|\n
                        |duration patient is in operating room|duration of anesthesia|total operation time") %>%
  filter(!str_detect(vars, "cm_icd") & !str_detect(vars, "dcnscva") & !str_detect(vars, "noprenafl")) 
maizlin_vars %>% print(n = 42)
nrow(maizlin_vars) # their table 1 has 42 variables contrary to what the methods say (43

maizlin_msng_long <- dat_all[c("operyr", maizlin_vars$vars)] %>%
  group_by(operyr) %>%
  miss_var_summary()

# wide format
maizlin_msng_long %>%
  select(-n_miss) %>%
  spread(operyr, pct_miss) %>%
  print(n = 50)

# visualizing missingness as a heatmap 
png(filename = paste0(figure_dir, "maizlin_var_msngness_yrs.png"), width = 640, height = 860)
maizlin_msng_long %>%
  ggplot(aes(x = operyr, y = variable)) + 
  geom_tile(aes(fill = pct_miss), colour = "black", alpha = 0.8) + 
  geom_label(aes(label = round(pct_miss, 2), fill = pct_miss)) + 
  scale_fill_viridis_c() +
  scale_x_continuous(breaks = seq(2012, 2018)) +
  ggtitle("Maizlin et al. Variable Missingness", "NSQIP-P 2012 - 2018") +
  labs(x = "Calendar Year", y = "Variable Name")
dev.off()


# create a new dataframe with their variables
maizlin_dat <- dat_all[c("operyr", maizlin_vars$vars)]  %>%
  filter(operyr <= 2014) %>%
  select(-c(prwbc, pralbum, proper30)) %>% # low missingness variables
  mutate(cpt = as.character(cpt))

# the numeric variables
maizlin_dat_num <- maizlin_dat %>%
  select_if(is.numeric) %>% 
  group_by(operyr)

# prior to imputation
maizlin_dat_num %>%
  group_by(operyr) %>% 
  miss_var_summary() %>%
  filter(n_miss > 0)

# after imputation
maizlin_dat_num_impute <- maizlin_dat_num %>%
  group_by(operyr) %>% 
  mutate_all(~ ifelse(is.na(.), mean(., na.rm = TRUE), .)) 
maizlin_dat_num_impute %>%
  miss_var_summary()
# looks good

# the factors
maizlin_dat_chr <- maizlin_dat %>%
  select_if(is.character) %>% 
  mutate(operyr = dat_all$operyr[dat_all$operyr <= 2014]) %>%
  group_by(operyr)

maizlin_dat_chr %>%  miss_var_summary() %>% filter(n_miss > 0) # doesn't need imputation


maizlin_dat_chr


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
  
  
 # join the raw 5 digit cpt codes to their general, surgical counterpart
# get all distinct codes
# i don't like the use of an external library for this but it seems the most simple way to do so

# if cpt code is within a surgery code range, then assign it accordingly
dat_cpt_unq <- dat_all %>% 
  ungroup() %>% 
  distinct(cpt) %>% 
  mutate(row_num = row_number()) %>% as_tibble()

dat_cpt_unq

cpt_cats <- sqldf("SELECT * FROM dat_cpt_unq
        LEFT JOIN surgery_cpt
        ON dat_cpt_unq.cpt BETWEEN surgery_cpt.code_start AND surgery_cpt.code_end") %>%
  as_tibble()
cpt_cats # mapping for all 716

cpt_cats$cpt <- as.character(cpt_cats$cpt)

# rejoin the low level cpt codes with the high level ones 
# cpt_cats$categories[which(cpt_cats$cpt == 11441)] # test

maizlin_dat_chr_compl <- maizlin_dat_chr %>% 
  left_join(select(cpt_cats, c(cpt, categories)), by = c("cpt")) %>%
  select(-c(operyr, cpt)) #%>%

# checking for variable freqs.
 maizlin_dat_chr_compl %>% 
  gather(key = "variable", value = "value", -one_of("operyr")) %>%
  count(variable, value) %>% 
  print(n = 300)
  
# since no missingness we now create the model matrix
maizlin_fct_x <- maizlin_dat_chr_compl[, -1] %>%
  model.matrix(data = ., ~ .)

dim(maizlin_fct_x) # 183233 x 66

# join the numeric with the character variables, adding back on the caseids

maizlin_x <- maizlin_fct_x[, -1] %>% 
  as_tibble() %>%
  bind_cols(., maizlin_dat_num_impute) %>%
  mutate(caseid = dat_all$caseid[dat_all$operyr <= 2014]) %>%
  select(caseid, operyr, everything()) 


maizlin_y <- mdl_maizlin %>% filter(operyr <= 2014)

length(maizlin_y)
dim(maizlin_x)

write_csv(maizlin_y, "output/maizlin_y.csv")
write_csv(maizlin_x, "output/maizlin_x.csv")



# ---- scratch ----
# if we wants the counts the 'wide' way
# overall outcomes 
# maizlin_outcomes_cts <-  maizlin_outcomes %>%
#   map_dfr(function(outcome_str){
#     print(outcome_str)
#     outcome <- ensym(outcome_str)
#     
#     
#     outcome_counts <- dat_all %>% 
#       count(!!outcome) %>%
#       mutate(perc= prop.table(n) * 100) %>%
#       mutate(total_n = sum(n)) %>%
#       rename(variable = !!outcome) #%>% 
#     
#     outcome_counts$var <- outcome_str
#     outcome_counts <- outcome_counts %>% select(var, variable, n, perc, total_n)
#     return(outcome_counts)
#     
#     # print(outcome_counts)
#   }) %>%
#   left_join(master_key, by = c("var" = "vars")) %>%
#   mutate(variable_label = str_replace_all(variable_label, "occurrences ", ""),
#          variable_label = str_replace_all(variable_label, "\\\r", " "))
# 
# maizlin_outcomes_cts

# the 'long' (and shorter! way)

# outcomes by cpt 
# would be interesting to see a pareto plot for each one
# cpt_type = "cpt_gen"
cpt_by_pos_class <- maizlin_outcomes %>%
  map_dfr(function(outcome ){ # cpt_gen or cpt
    print(outcome)
    outcome <- ensym(outcome)
    # cpt_type <- ensym(cpt_type)
    outcome_counts <- dat_all %>% 
      # count outcomes by general cpt code
      count(!!outcome, cpt, sort = TRUE) %>%
      filter(!!outcome != "no complication") %>% # surprisingly faster to count then filter out as opposed to the other way around
      # count(!!outcome) %>%
      mutate(perc= prop.table(n) * 100) %>%
      rename(variable = !!outcome) %>%
      # get total counts of the class
      group_by(variable) %>% mutate(total_n = sum(n)) #%>%
    # mutate(sum = )
    # convert to wide
    # select(-n) %>%
    # spread(cpt_gen, perc)
    # 
    # print(outcome_counts %>%
    #   ggplot(aes(x = log(n))) +
    #   geom_histogram() )
    
    print(outcome_counts)
  }) 
# convert this to long when i have time - DONE


# what was i doing here 
# dat_all %>% count(cpt) %>% arrange(cpt) 
# dat_all %>% count(cpt, prncptx) %>% arrange(cpt) %>% 
#   group_by(cpt) %>% fill(prncptx) %>%
#   group_by(cpt, prncptx) %>%
#   summarize(n = sum(n)) %>%
#   print(n = 100)
#(?)

