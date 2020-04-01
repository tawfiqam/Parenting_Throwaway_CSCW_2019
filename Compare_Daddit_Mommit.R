---
  title: "R Notebook"
output: html_notebook
---
  
  This is an [R Markdown](http://rmarkdown.rstudio.com) Notebook. When you execute code within the notebook, the results appear beneath the code. 

Try executing this chunk by clicking the *Run* button within the chunk or by placing your cursor inside it and pressing *Cmd+Shift+Enter*. 

```{r}
plot(cars)
```

Add a new chunk by clicking the *Insert Chunk* button on the toolbar or by pressing *Cmd+Option+I*.

When you save the notebook, an HTML file containing the code and output will be saved alongside it (click the *Preview* button or press *Cmd+Shift+K* to preview the HTML file). 

The preview shows you a rendered HTML copy of the contents of the editor. Consequently, unlike *Knit*, *Preview* does not run any R code chunks. Instead, the output of the chunk when it was last run in the editor is displayed.

options(stringsAsFactors = FALSE)
library(tm)
require("SnowballC")
require(slam)
library(wordcloud)   

english_stopwords <- stopwords("en")

lines <- readLines("Daddit.txt", encoding = "UTF-8")
targetCorpus <- Corpus(VectorSource(lines))
targetCorpus <- tm_map(targetCorpus, removePunctuation, preserve_intra_word_dashes = TRUE)
targetCorpus <- tm_map(targetCorpus, removeNumbers)
targetCorpus <- tm_map(targetCorpus, content_transformer(tolower))
targetCorpus <- tm_map(targetCorpus, removeWords, english_stopwords)
targetCorpus <- tm_map(targetCorpus, stemDocument, language = "en")
targetCorpus <- tm_map(targetCorpus, stripWhitespace)
targetCorpus <- tm_map(targetCorpus, content_transformer(removePunctuation))

targetDTM <- DocumentTermMatrix(targetCorpus)
termCountsTarget <- col_sums(targetDTM)

lines <- readLines("Mommit.txt", encoding = "UTF-8")
comparisonCorpus <- Corpus(VectorSource(lines))
comparisonCorpus <- tm_map(comparisonCorpus, removePunctuation, preserve_intra_word_dashes = TRUE)
comparisonCorpus <- tm_map(comparisonCorpus, removeNumbers)
comparisonCorpus <- tm_map(comparisonCorpus, content_transformer(tolower))
comparisonCorpus <- tm_map(comparisonCorpus, removeWords, english_stopwords)
comparisonCorpus <- tm_map(comparisonCorpus, stemDocument, language = "en")
comparisonCorpus <- tm_map(comparisonCorpus, stripWhitespace)
comparisonCorpus <- tm_map(comparisonCorpus, content_transformer(removePunctuation))

comparisonDTM <- DocumentTermMatrix(comparisonCorpus)
termCountsComparison <- col_sums(comparisonDTM)

# Loglikelihood for a single term
term <- "care"

# Determine variables
a <- termCountsTarget[term]
b <- termCountsComparison[term]
c <- sum(termCountsTarget)
d <- sum(termCountsComparison)

Expected1 = c * (a+b) / (c+d)
Expected2 = d * (a+b) / (c+d)
t1 <- a * log((a/Expected1))
t2 <- b * log((b/Expected2))
logLikelihood <- 2 * (t1 + t2)

print(logLikelihood)

# use set operation to get terms only occurring in target document
uniqueTerms <- setdiff(names(termCountsTarget), names(termCountsComparison))
# Have a look into a random selection of terms unique in the target corpus
sample(uniqueTerms, 20)

# Create vector of zeros to append to comparison counts
zeroCounts <- rep(0, length(uniqueTerms))
names(zeroCounts) <- uniqueTerms
termCountsComparison <- c(termCountsComparison, zeroCounts)

# Get list of terms to compare from intersection of target and comparison vocabulary
termsToCompare <- intersect(names(termCountsTarget), names(termCountsComparison))

# Calculate statistics (same as above, but now with vectors!)
a <- termCountsTarget[termsToCompare]
b <- termCountsComparison[termsToCompare]
c <- sum(termCountsTarget)
d <- sum(termCountsComparison)
Expected1 = c * (a+b) / (c+d)
Expected2 = d * (a+b) / (c+d)
t1 <- a * log((a/Expected1) + (a == 0))
t2 <- b * log((b/Expected2) + (b == 0))
logLikelihood <- 2 * (t1 + t2)

# Compare relative frequencies to indicate over/underuse
relA <- a / c
relB <- b / d
# underused terms are multiplied by -1
logLikelihood[relA < relB] <- logLikelihood[relA < relB] * -1

sort(logLikelihood, decreasing=TRUE)[1:100]

sort(logLikelihood, decreasing=FALSE)[1:100]

sorted_loglikelihood_increase <- sort(logLikelihood, decreasing=TRUE)
sorted_loglikelihood_decrease <- sort(logLikelihood, decreasing=FALSE)

write.table(sorted_loglikelihood_increase, "loglikelihood_All_Throw_Conv.csv", row.names = TRUE, col.names=TRUE, sep = "\t")
write.table(sorted_loglikelihood_decrease, "loglikelihood_All_Pseud_Conv.csv", row.names = TRUE, col.names=TRUE, sep = "\t")