library(syuzhet)
library(ggplot2)
library(tidyr)
my_example_text <- "I begin this story with a neutral statement.  
Basically this is a very silly test.  
You are testing the Syuzhet package using short, inane sentences.  
I am actually very happy today. 
I have finally finished writing this package.  
Tomorrow I will be very sad. 
I won't have anything left to do. 
I might get angry and decide to do something horrible.  
I might destroy the entire package and start from scratch.  
Then again, I might find it satisfying to have completed my first R package. 
Honestly this use of the Fourier transformation is really quite elegant.  
You might even say it's beautiful!"
s_v <- get_sentences(my_example_text)

syuzhet_vector <- get_sentiment(s_v, method="syuzhet")
bing_vector <- get_sentiment(s_v, method="bing")
afinn_vector <- get_sentiment(s_v, method="afinn")
nrc_vector <- get_sentiment(s_v, method="nrc")

rbind(
  sign(syuzhet_vector),
  sign(bing_vector),
  sign(afinn_vector),
  sign(nrc_vector)
)

all_methods <- as.data.frame(cbind(syuzhet_vector, 
                                   bing_vector, 
                                   afinn_vector, 
                                   nrc_vector,
                                   sent = 1:12))

# Simple rescaling of y axis from -1 to 1
# To rescale also the x axis use rescale_x_2
rescaled_all = data.frame(all_methods[5], apply(all_methods[1:4],2, rescale) )
rescaled_all %>%
  gather(key,value, syuzhet_vector, bing_vector, afinn_vector, nrc_vector) %>%
  ggplot(aes(x=sent, y=value, colour=key)) +
  geom_line()



# simple plot - 3 types of smoothing applied
# Second plot shows a discrete cosine transformation (DCT) of the values
# It’s main advantage is in its better representation of edge values in 
# the smoothed version of the sentiment vector
simple_plot(syuzhet_vector)
simple_plot(bing_vector)
simple_plot(afinn_vector)
simple_plot(nrc_vector)


# Euclidean distance bwtween methods (0 indicates perfect match)
dist(t(rescaled_all[2:5]))

# Correlations
cor(rescaled_all[2:5])

nrc_data <- get_nrc_sentiment(s_v)
barplot(
  sort(colSums(prop.table(nrc_data[, 1:8]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  las = 1, 
  main = "Emotions in Sample text", xlab="Percentage"
)



