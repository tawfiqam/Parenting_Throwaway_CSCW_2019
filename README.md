#  How Self-declared Throwaway Accounts Enable Parenting Disclosure and Support on Reddit
This repository shares the notebooks we used in our analysis of throwaway accounts in parenting subreddits. Note that the notebooks
do not always render successfully on github. If the notebook does not render on github, you can open it by copying the url and posting 
it [here](https://nbviewer.jupyter.org/).

In our [CSCW paper](http://www-personal.umich.edu/~tawfiqam/CSCW_2019_Reddit_Ammari.pdf), titled "How Self-declared Throwaway Accounts Enable Parenting Disclosureand Support on Reddit,"
we drew on 10 years of Reddit data from the three main parenting subreddits to answer the following research questions:
- **RQ1:** What are the predictors of parents posting to Reddit as throwaways?
- **RQ2:** What are the main themes discussed by throwaways?
- **RQ3:** How do the responses to throwaway comments differ from responses to other comments?

## Dataset
We used a publicly available Reddit dataset. This dataset was collected by [Baumgartner using the Reddit API](https://files.pushshift.io/reddit/comments/). The data we use in our analysis were drawn from public subreddits between March 31st of 2008 and October 31st of 2018. 

## RQ1: Predictors of posting using throwaway accounts
Most of this analysis can be seen in [this notebook](https://github.com/tawfiqam/Parenting_Throwaway_CSCW_2019/blob/master/TF_IDF_LLR_Word2Vec_Parenting_After_Models_Final.ipynb)

**Logistic regression classifier**

We used the [SKLearn Logistic Regression classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) with L2 regularization in order to predict whether a throwaway account will be used. The class we analyzed in our logistic regression classifier, throwaway comments, is the minority sample in the dataset. In order to balance the dataset, we under-sampled the majority class using [](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html). Undersampling balances the dataset by randomly removing values from the majority dataset (non-throwaway comments). This generated a 50:50 class ratio for the classifier with a baseline accuracy of 0.5.

**Features for our classifier**

- LDA Topic Modeling for Topic Detection [60 features] <br/>
We used [Gensim](https://radimrehurek.com/gensim/index.html) to train 9 LDA models starting with a number of 
topics k=10, with a step of 10 topics until a limit of 90 topics. 
For each of these k iterations, we calculated the coherence of 
the LDA models using the gensim [CoherenceModel](https://radimrehurek.com/gensim/models/coherencemodel.html) feature from Gensim. The code for this analysis can be found in [this script](https://github.com/tawfiqam/Parenting_Throwaway_CSCW_2019/blob/master/ReadLDA_Model.py).

- LIWC linguistic measures [72 features] <br/>
We used the Linguistic Inquiry and Word Count (LIWC) text analysis program, a lexicon of linguistic categories that has been psycho-metrically validated and performs well on social media data sets to extract lexico-syntactic features. We applied LIWC 2015 processor on each of the comments in our dataset.

- Control features [3 features] <br/>
In addition to the 60 LDA topic features and 72 LIWC linguistic categories, we also used 3 control features: (1) average user tenure; (2) average Karma score per comment; and (3) average comment length.

## RQ2: Main themes discussed by throwaways
To contextualize results from study 1 above, we expand on our quantitative results using both quantitative and qualitative methods. We use Log Likelihood Ratio (LLR) to develop themes associated with the LDA topics, and use qualitative analysis to check and expand our inquiry into the nature of parents’ comments.

The Log Likelihood Ratio is the logarithm of the ratio of the probability of the word’s occurrence in throwaway comments to the probability of it occurring in pseudonymous comments. LLR analysis requires two documents to compare. In our case, LLR is used to compare throwaway and pseudonymous conversation discussing each of the significant LDA topics. The script we used to compute LLRs can be found [here](https://github.com/tawfiqam/Parenting_Throwaway_CSCW_2019/blob/master/LLR.rmd).

## RQ3: How do the responses to throwaway comments differ from responses to other comments?
We draw on methods from causal analysis to calculate the effect of the treatment (using a throwaway account) to the outcome (change in number of posts, score, number of posts etc.) while controlling for the effects of LDA topics and LIWC categories to reduce bias based on the confounding variables (determined in study 1). Specifically, we employ Propensity Score Matching [PSM](https://en.wikipedia.org/wiki/Propensity_score_matching).

We used logistic regression on the covariates to calculate propensity scores for our PSM. We then matched the throwaway and pseudonymous groups using 1:1 nearest neighbor matching. We used a nearest neighbor (KNN) algorithm with a caliper of 0.05 - matching on the logit of the propensity score using calipers of width equal to 0.05 of the standard deviation of the logit of the propensity score. The script can be found [here](https://github.com/tawfiqam/Parenting_Throwaway_CSCW_2019/blob/master/PSM_Final_Normalized.ipynb). 

