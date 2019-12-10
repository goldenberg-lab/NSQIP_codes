library(ggplot2)
library(dplyr)

# ---- set-up ----
theme_set(theme_minimal(base_size = 14) +
            theme(plot.title = element_text(hjust = 0.5),
                  plot.subtitle = element_text(hjust = 0.5)))

get_def <- function(term, data = master_key ){
  "
  
  :param term: (str)a variables description to search by
  :return: (str) matching variable name(s)
  
  
  example: 
  get_def('pneumonia')
  
    # A tibble: 4 x 2
    vars      variable_label                     
    <chr>     <chr>                              
  1 cpneumon  current pneumonia                  
  2 noupneumo number of pneumonia occurrences    
  3 oupneumo  occurrences pneumonia              
  4 doupneumo days from operation until pneumonia
  
  "
  
  res <- data %>% 
    filter(str_detect(variable_label,
                      !!term))
  
  return(res)}


get_def(term = "open wound|diabetes|premature|ventilator|current pneumonia")
