"""
first download all of the files onto local computer via git pull
"""


##Loading the data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import glob
import json


#root_path = '/kaggle/input/CORD-19-research-challenge/'
root_path = '../input/a-custom-literature-corpus'
metadata_path = '../input/a-custom-literature-corpus/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'doi': str, 
    'title': str,
    'authors': str,
})


meta_df.head()

meta_df.info()

##Fetch all of JSON file path
root_path = '../input/possibilities'
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True) 
len(all_json)

##checking JSON Schema Structure
with open(all_json[0]) as file:
    first_entry = json.load(file)
    print(json.dumps(first_entry, indent=4))
    
#Helper: File Reader Class
class FileReader:
"""
Helper function adds break after every words when character length reach 
to certain amount. This is for the interactive plot so that hover tool fits the screen.
"""
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
           # self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content["abstract"]:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['full_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
            # Extend Here
            #
            #
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)

def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data
    
##Load the data into dataframe Using the helper functions, let's read in the articles into a DataFrame that can be used easily:

dict_ = {'paper_id': [], 'doi':[], 'abstract': [], 'full_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    
    try:
        content = FileReader(entry)
    except Exception as e:
        continue  # invalid paper format, skip
    
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    
    dict_['abstract'].append(content.abstract)
    dict_['paper_id'].append(content.paper_id)
    dict_['full_text'].append(content.body_text)
    
    # also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 0: 
        # no abstract provided
        dict_['abstract'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 100:
        # abstract provided is too long for plot, take first 100 words append with ...
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        dict_['abstract'].append(summary + "...")
    else:
        # abstract is short enough
        summary = get_breaks(content.abstract, 40)
        dict_['abstract'].append(summary)
        
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    
    try:
        # if more than one author
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            # if more than 2 authors, take them all with html tag breaks in between
            dict_['authors'].append(get_breaks('. '.join(authors), 40))
        else:
            # authors will fit in plot
            dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # if only one author - or Null valie
        dict_['authors'].append(meta_data['authors'].values[0])
    
    # add the title information, add breaks when needed
    try:
        title = get_breaks(meta_data['title'].values[0], 40)
        dict_['title'].append(title)
    # if title was not provided
    except Exception as e:
        dict_['title'].append(meta_data['Title'].values[0])
    
    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])
    
    # add doi
    dict_['doi'].append(meta_data['doi'].values[0])
    
df_reef = pd.DataFrame(dict_, columns=['paper_id', 'doi', 'abstract', 'full_text', 'authors', 'title', 'journal', 'abstract_summary'])
df_reef.head()

#Clean duplicates There may be duplicates by author submiting the article to multiple journals
df_reef.drop_duplicates(['Abstract', 'Full_text'], inplace=True)
df_reef['Abstract'].describe(include='all')

df_reef['Full_text'].describe(include='all')


#Look at data
df_reef.head()

#In the majority of this notebook we will be working with body_text Links to the papers will be generated using doi
df_reef.describe()

#Data preprocessing Kaggle limits the dataframe to 10,000 instances
df = df_reef.sample(10000, random_state=42)
del df_reef

#Now that we have our dataset loaded, we need to clean-up the text to improve any clustering or classification efforts. First, drop Null vales
df.dropna(inplace=True)
df.info()


#Download SpaCy bio parser with pip
from IPython.utils import io
with io.capture_output() as captured:
     pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
  
#NLP 
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_sci_lg  # model downloaded in previous step

#Stopwords Part of the preprocessing will be finding and removing stopwords (common words that will act as noise in the clustering step).
punctuations = string.punctuation
stopwords = list(STOP_WORDS)
stopwords[:10]

"""
Now the above stopwords are used in everyday english text. 
Research papers will often frequently use words that don't actually contribute to the meaning 
and are not considered everyday stopwords.
"""
custom_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 
    'al.', 'Elsevier', 'PMC', 'CZI', 'www'
]

for w in custom_stop_words:
    if w not in stopwords:
        stopwords.append(w)
        
"""
Next let's create a function that will process the text data for us 
For this purpose we will be using the spacy library. This function will convert 
text to lower case, remove punctuation, and find and remove stopwords. For the parser, 
we will use en_core_sci_lg. This is a model for processing biomedical, scientific or clinical text.
"""
# Parser
parser = en_core_sci_lg.load(disable=["tagger", "ner"])
parser.max_length = 7000000

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens
    
#Apply the text-processing function on the body_text
tqdm.pandas()
df["processed_text"] = df["Full_text"].progress_apply(spacy_tokenizer)

"""
Vectorization Now that we have pre-processed the data, it is time to convert 
it into a format that can be handled by our algorithms. For this purpose we will 
be usin tf-idf. This will convert our string formatted data into a measure of how 
important each word is to the instance out of the literature as a whole.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize(text, maxx_features):
    
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(text)
    return X
    
"""
Vectorize our data. We will be clustering based off the content 
of the body text. The maximum number of features will be limited. 
Only the top 2 ** 12 features will be used, eseentially acting as a 
noise filter. Additionally, more features cause painfully long runtimes.
"""
text = df['processed_text'].values
X = vectorize(text, 2 ** 12)
X.shape

"""PCA & Clustering Let's see how much we can reduce the dimensions while still 
keeping 95% variance. We will apply Principle Component Analysis (PCA) to our 
vectorized data. The reason for this is that by keeping a large number of dimensions 
with PCA, you don’t destroy much of the information, but hopefully will remove some 
noise/outliers from the data, and make the clustering problem easier for k-means. Note 
that X_reduced will only be used for k-means, t-SNE will still use the original feature 
vector X that was generated through tf-idf on the NLP processed text.
"""
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)
X_reduced= pca.fit_transform(X.toarray())
X_reduced.shape


"""
To separate the literature, k-means will be run on the vectorized text. Given the number 
of clusters, k, k-means will categorize each vector by taking the mean distance to a randomly 
initialized centroid. The centroids are updated iteratively.
"""
from sklearn.cluster import KMeans

Image(filename='resources/kmeans.png', width=800, height=800)


"""
How many clusters?
To find the best k value for k-means we'll look at the distortion at different k values. 
Distortion computes the sum of squared distances from each point to its assigned center. 
When distortion is plotted against k there will be a k value after which decreases in distortion 
are minimal. This is the desired number of clusters.
"""
from sklearn import metrics
from scipy.spatial.distance import cdist

# run kmeans with many different k
distortions = []
K = range(2, 50)
for k in K:
    k_means = KMeans(n_clusters=k, random_state=42, n_jobs=-1).fit(X_reduced)
    k_means.fit(X_reduced)
    distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    #print('Found distortion for {} clusters'.format(k))
    
X_line = [K[0], K[-1]]
Y_line = [distortions[0], distortions[-1]]

# Plot the elbow
plt.plot(K, distortions, 'b-')
plt.plot(X_line, Y_line, 'r')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

"""
In this plot we can see that the better k values are between 18-25. After that, the decrease in 
distortion is not as significant. For simplicity, we will use k=20
"""
"""
Run k-means¶
Now that we have an appropriate k value, we can run k-means on the PCA-processed feature vector (X_reduced).
"""
k = 20
kmeans = KMeans(n_clusters=k, random_state=42, n_jobs=-1)
y_pred = kmeans.fit_predict(X_reduced)
df['y'] = y_pred

"""
Dimensionality Reduction with t-SNE
Using t-SNE we can reduce our high dimensional features vector to 2 dimensions. By using the 2 dimensions as x,y coordinates, the body_text can be plotted.

t-Distributed Stochastic Neighbor Embedding (t-SNE) reduces dimensionality while trying to keep similar instances close and dissimilar instances apart. It is mostly used for visualization, in particular to visualize clusters of instances in high-dimensional space

Cite: Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow: Second Edition | Aurélien Geron
"""


from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=100, random_state=42)
X_embedded = tsne.fit_transform(X.toarray())

"""
So that step took a while! Let's take a look at what our data looks like when compressed to 2 dimensions.
"""

from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", 1)

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)
plt.title('t-SNE with no Labels')
plt.savefig("plots/t-sne_covid19.png")
plt.show()


"""
This looks pretty bland. There are some clusters we can immediately detect, but the many 
instances closer to the center are harder to separate. t-SNE did a good job at reducing the 
dimensionality, but now we need some labels. Let's use the clusters found by k-means as labels. 
This will help visually separate different concentrations of topics.
"""

%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(15, 15)})

# colors
palette = sns.hls_palette(20, l=.4, s=.9)

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title('t-SNE with Kmeans Labels')
plt.savefig("plots/improved_cluster_tsne.png")
plt.show()

"""
The location of each paper on the plot was determined by t-SNE while the label (color) was determined by k-means. 
If we look at a particular part of the plot where t-SNE has grouped many articles forming a cluster, it is likely 
that k-means is uniform in the labeling of this cluster (most of the cluster is the same color). This behavior 
shows that structure within the literature can be observed and measured to some extent
"""

"""
Topic Modeling on Each Cluster
Now we will attempt to find the most significant words in each clusters. K-means clustered the 
articles but did not label the topics. Through topic modeling we will find out what the most 
important terms for each cluster are. This will add more meaning to the cluster by giving keywords 
to quickly identify the themes of the cluster.

For topic modeling, we will use LDA (Latent Dirichlet Allocation). In LDA, each document can be 
described by a distribution of topics and each topic can be described by a distribution of words.
"""

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
Image(filename='resources/lda.jpg', width=600, height=600)


#First we will create 20 vectorizers, one for each of our cluster labels

vectorizers = []
    
for ii in range(0, 20):
    # Creating a vectorizer
    vectorizers.append(CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}'))
vectorizers[0]


#Now we will vectorize the data from each of our clusters

vectorized_data = []

for current_cluster, cvec in enumerate(vectorizers):
    try:
        vectorized_data.append(cvec.fit_transform(df.loc[df['y'] == current_cluster, 'processed_text']))
    except Exception as e:
        print("Not enough instances in cluster: " + str(current_cluster))
        vectorized_data.append(None)
len(vectorized_data)

#Topic modeling will be performed through the use of Latent Dirichlet Allocation 
#(LDA). This is a generative statistical model that allows sets of words to be explained by a shared topic

# number of topics per cluster
NUM_TOPICS_PER_CLUSTER = 20

lda_models = []
for ii in range(0, 20):
    # Latent Dirichlet Allocation Model
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS_PER_CLUSTER, max_iter=10, learning_method='online',verbose=False, random_state=42)
    lda_models.append(lda)
    
lda_models[0]

#For each cluster, we had created a correspoding LDA model in the previous step. We will now fit_transform all the 
#LDA models on their respective cluster vectors
clusters_lda_data = []

for current_cluster, lda in enumerate(lda_models):
    # print("Current Cluster: " + str(current_cluster))
    
    if vectorized_data[current_cluster] != None:
        clusters_lda_data.append((lda.fit_transform(vectorized_data[current_cluster])))
        
"""
Extracts the keywords from each cluster
"""

# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=3):
    current_words = []
    keywords = []
    
    for idx, topic in enumerate(model.components_):
        words = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])
                
    keywords.sort(key = lambda x: x[1])  
    keywords.reverse()
    return_values = []
    for ii in keywords:
        return_values.append(ii[0])
    return return_values
"""
Append list of keywords for a single cluster to 2D list of length NUM_TOPICS_PER_CLUSTER
"""

all_keywords = []
for current_vectorizer, lda in enumerate(lda_models):
    # print("Current Cluster: " + str(current_vectorizer))

    if vectorized_data[current_vectorizer] != None:
        all_keywords.append(selected_topics(lda, vectorizers[current_vectorizer]))
all_keywords[0][:10]        

len(all_keywords)

"""
Save current outputs to file
Re-running some parts of the notebook (especially vectorization and t-SNE) are time intensive tasks. 
We want to make sure that the important outputs for generating the bokeh plot are saved for future use.
"""

f=open('lib/topics.txt','w')

count = 0

for ii in all_keywords:

    if vectorized_data[count] != None:
        f.write(', '.join(ii) + "\n")
    else:
        f.write("Not enough instances to be determined. \n")
        f.write(', '.join(ii) + "\n")
    count += 1

f.close()
import pickle

# save the COVID-19 DataFrame, too large for github
pickle.dump(df, open("plot_data/df_covid.p", "wb" ))

# save the final t-SNE
pickle.dump(X_embedded, open("plot_data/X_embedded.p", "wb" ))

# save the labels generate with k-means(20)
pickle.dump(y_pred, open("plot_data/y_pred.p", "wb" ))

"""
Classify
Though arbitrary, after running kmeans, the data is now 'labeled'. This means that we now 
use supervised learning to see how well the clustering generalizes. This is just one way 
to evaluate the clustering. If k-means was able to find a meaningful split in the data, it 
should be possible to train a classifier to predict which cluster a given instance should belong to.
"""

# function to print out classification model report
def classification_report(model_name, test, pred):
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    
    print(model_name, ":\n")
    print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(test, pred)) * 100), "%")
    print("     Precision: ", '{:,.3f}'.format(float(precision_score(test, pred, average='macro')) * 100), "%")
    print("        Recall: ", '{:,.3f}'.format(float(recall_score(test, pred, average='macro')) * 100), "%")
    print("      F1 score: ", '{:,.3f}'.format(float(f1_score(test, pred, average='macro')) * 100), "%")
    
"""
Let's split the data into train/test sets
"""

from sklearn.model_selection import train_test_split

# test set size of 20% of the data and the random seed 42 <3
X_train, X_test, y_train, y_test = train_test_split(X.toarray(),y_pred, test_size=0.2, random_state=42)

print("X_train size:", len(X_train))
print("X_test size:", len(X_test), "\n")


"""
Now let's create a Stochastic Gradient Descent classifier

Precision is ratio of True Positives to True Positives + False Positives. This is the accuracy of positive predictions
Recall (also known as TPR) measures the ratio of True Positives to True Positives + False Negatives. It measures the 
ratio of positive instances that are correctly detected by the classifer.
F1 score is the harmonic average of the precision and recall. F1 score will only be high if both precision and recall are high

Cite: Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow: Second Edition | Aurélien Geron
"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier

# SGD instance
sgd_clf = SGDClassifier(max_iter=10000, tol=1e-3, random_state=42, n_jobs=-1)
# train SGD
sgd_clf.fit(X_train, y_train)

# cross validation predictions
sgd_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3, n_jobs=-1)

# print out the classification report
classification_report("Stochastic Gradient Descent Report (Training Set)", y_train, sgd_pred)

 
"""
To test for overfitting, let's see how the model generalizes over the test set
"""

# cross validation predictions
sgd_pred = cross_val_predict(sgd_clf, X_test, y_test, cv=3, n_jobs=-1)

# print out the classification report
classification_report("Stochastic Gradient Descent Report (Training Set)", y_test, sgd_pred)

"""
Now let's see how the model can generalize across the whole dataset.
"""
sgd_cv_score = cross_val_score(sgd_clf, X.toarray(), y_pred, cv=10)
print("Mean cv Score - SGD: {:,.3f}".format(float(sgd_cv_score.mean()) * 100), "%")

"""
Plotting the data
The previous steps have given us clustering labels and a dataset of papers reduced to 
two dimensions. By pairing this with Bokeh, we can create an interactive plot of the literature. 
This should organize the papers such that related publications are in close proximity. To try to 
undertstand what the similarities may be, we have also performed topic modelling on each cluster 
of papers in order to pick out the key terms.
Bokeh will pair the actual papers with their positions on the t-SNE plot. Through this approach 
it will be easier to see how papers fit together, allowing for both exploration of the dataset 
and evaluation of the clustering.
"""


# required libraries for plot
from lib.plot_text import header, description, description2, cite, description_search, description_slider, notes, dataset_description, toolbox_header 
from lib.call_backs import input_callback, selected_code
import bokeh
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS, Slider, TapTool, TextInput
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap, transform
from bokeh.io import output_file, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import RadioButtonGroup, TextInput, Div, Paragraph
from bokeh.layouts import column, widgetbox, row, layout
from bokeh.layouts import column

"""
Load the Keywords per Cluster
"""
import os

topic_path = os.path.join(os.getcwd(), 'lib', 'topics.txt')
with open(topic_path) as f:
    topics = f.readlines()
"""
Setup
"""

# show on notebook
output_notebook()
# target labels
y_labels = y_pred

# data sources
source = ColumnDataSource(data=dict(
    x= X_embedded[:,0], 
    y= X_embedded[:,1],
    x_backup = X_embedded[:,0],
    y_backup = X_embedded[:,1],
    desc= y_labels, 
    titles= df['title'],
    authors = df['authors'],
    journal = df['journal'],
    abstract = df['abstract_summary'],
    labels = ["C-" + str(x) for x in y_labels],
    links = df['doi']
    ))

# hover over information
hover = HoverTool(tooltips=[
    ("Title", "@titles{safe}"),
    ("Author(s)", "@authors{safe}"),
    ("Journal", "@journal"),
    ("Abstract", "@abstract{safe}"),
    ("Link", "@links")
],
point_policy="follow_mouse")

# map colors
initial_palette = Category20[20]
random.Random(42).shuffle(initial_palette)

mapper = linear_cmap(field_name='desc', 
                     palette=Category20[20],
                     low=min(y_labels) ,high=max(y_labels))

# prepare the figure
plot = figure(plot_width=1200, plot_height=850, 
           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'tap'], 
           title="Clustering of the COVID-19 Literature with t-SNE and K-Means", 
           toolbar_location="above")

# plot settings
plot.scatter('x', 'y', size=5, 
          source=source,
          fill_color=mapper,
          line_alpha=0.3,
          line_width=1.1,
          line_color="black",
          legend = 'labels')
plot.legend.background_fill_alpha = 0.6
 BokehJS 1.3.4 successfully loaded.
 
""" 
Widgets
"""
# Keywords
text_banner = Paragraph(text= 'Keywords: Slide to specific cluster to see the keywords.', height=25)
input_callback_1 = input_callback(plot, source, text_banner, topics)

# currently selected article
div_curr = Div(text="""Click on a plot to see the link to the article.""",height=150)
callback_selected = CustomJS(args=dict(source=source, current_selection=div_curr), code=selected_code())
taptool = plot.select(type=TapTool)
taptool.callback = callback_selected

# WIDGETS
slider = Slider(start=0, end=20, value=20, step=1, title="Cluster #", callback=input_callback_1)
keyword = TextInput(title="Search:", callback=input_callback_1)

# pass call back arguments
input_callback_1.args["text"] = keyword
input_callback_1.args["slider"] = slider
# column(,,widgetbox(keyword),,widgetbox(slider),, notes, cite, cite2, cite3), plot


"""
Style
"""
# STYLE
header.sizing_mode = "stretch_width"
header.style={'color': '#2e484c', 'font-family': 'Julius Sans One, sans-serif;'}
header.margin=5

description.style ={'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
description.sizing_mode = "stretch_width"
description.margin = 5

description2.sizing_mode = "stretch_width"
description2.style ={'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
description2.margin=10

description_slider.style ={'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
description_slider.sizing_mode = "stretch_width"

description_search.style ={'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
description_search.sizing_mode = "stretch_width"
description_search.margin = 5

slider.sizing_mode = "stretch_width"
slider.margin=15

keyword.sizing_mode = "scale_both"
keyword.margin=15

div_curr.style={'color': '#BF0A30', 'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
div_curr.sizing_mode = "scale_both"
div_curr.margin = 20

text_banner.style={'color': '#0269A4', 'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
text_banner.sizing_mode = "scale_both"
text_banner.margin = 20

plot.sizing_mode = "scale_both"
plot.margin = 5

dataset_description.sizing_mode = "stretch_width"
dataset_description.style ={'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
dataset_description.margin=10

notes.sizing_mode = "stretch_width"
notes.style ={'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
notes.margin=10

cite.sizing_mode = "stretch_width"
cite.style ={'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
cite.margin=10

r = row(div_curr,text_banner)
r.sizing_mode = "stretch_width"

"""
SHOW
"""
# LAYOUT OF THE PAGE
l = layout([
    [header],
    [description],
    [description_slider, description_search],
    [slider, keyword],
    [text_banner],
    [div_curr],
    [plot],
    [description2, dataset_description, notes, cite],
])
l.sizing_mode = "scale_both"


# show
output_file('plots/t-sne_covid-19_interactive.html')
show(l)

