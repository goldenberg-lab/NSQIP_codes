
library(dplyr)
library(purrr)
# Delvin So 
# 01a_keys 
# input: Manually extracted keys from  tables found in yearly UserGuides (why are they even tables in a pdf anyways??)
# returns: a single, master key of all variables and their descriptions found across NSQIP-P

# ---- set-up ----
# setwd('~/Documents/nsqip/')
setwd("~/Documents/OneDrive - SickKids/nsqip/") # or wherever the root nsqip directory is

# ---- read in the data ----
# location of the tabula extracted keys
dicts <- list.files(path = 'nsqip_extracted_keys/', full.names = TRUE)

# the variable names across the years
yr_vars_all <- read_csv("output/var_names_yrs.csv")

# # test with one
# y18_keys <- read_csv(dicts[length(dicts)]) %>%
#   mutate_all(~ str_to_lower(.)) %>%
#   rename_all(. %>% str_to_lower %>% str_replace_all("\\s", "_"))
# 
# y18_keys %>% filter(!is.na(variable_name))

# extract all the variable names and descriptions across each years key and wrangling for consistency
nsqip_key_vars <- dicts %>% 
  map_dfr(~ read_csv(.x) %>%
                mutate_all(~ str_to_lower(.)) %>%
                rename_all(. %>% str_to_lower %>% str_replace_all("\\s", "_"))) %>%
  filter(!is.na(variable_name), !is.na(variable_label)) %>%
  distinct(variable_name, variable_label) 

# all the variable names
yr_vars_all %>% distinct(vars) %>% nrow() # 399

# join on dictionary but only take first definition of each variable 
# this is because variable labels may not have been extracted cleanly and thus may be inconsistent across the years
# this also ensures we have a 1:1 correspondence between a key and its value
master_key <- yr_vars_all %>% distinct(vars) %>% # 389
  left_join( nsqip_key_vars,
             by = c("vars" = "variable_name")) %>% 
  # na.omit() %>%
  group_by(vars) %>%
  # arrange(variable_label) %>% # 
  filter(row_number() == 1) %>% print(n = 400)

# ---- save down master key ----
# write_csv(master_key, "output/master_key.csv")
  




