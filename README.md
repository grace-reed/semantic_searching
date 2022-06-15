
**Project Outline**
To develop a comparative diagnostic testing platform for upper respiratory viruses as well as gastrointestinal pathogens, using a golden standard as MiSeq with PCR against ONT library preparation in four phases.
### Phases ###



# Approach

- Parse the text from the body of each document using NLP
- Turn each document instance di into a feature vector Xi using Term-Frequency-inverse Document Frequency (TF-IDF)
- Apply dimentionality Reduction to each feature vector Xi using t-Distributed Stochastic Neighbor Embedding (t-SNE) to cluster similar research articles in the two dimentional plane X embedding Yi
- Use PCA t project down the dimentions of X to a number of dimentions that will keep 0.95 variance while removing noise and outliers in embedding Y2
- Apply k-means clustering on Y2 where K is 20, to label each clustern on Y1
- Apply Topic Modeling on X using Latent Dirichlet Allocation (LDA) to discover keywords for ech cluster
- Investigate the clusters wisually on the plot, zooming down to specific articles as needed, and via classification using Stochastic Gradient Descent (SGD)
- Credits for the literature clustering approach go to Eren, E. Maksim. Solovyev, Nick. Nicholas, Charles. Raff, Edward and their full interactive plot can be found on GitHub https://maksimekin.github.io/COVID19-Literature-Clustering/plots/t-sne_covid-19_interactive.html
-----------------------------------
Table of contents


1 loading the data

2 pre-processing

3 vectorization

4 PCA & Clustering

5 dientionality reduction with t-SNE

6 topic modeling on each cluster

7 classify

8 plot

8b think about how to use the plot

9 conclusion
