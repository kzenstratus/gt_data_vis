# CSE6242-Project-Visioneers
Project folder for CSE6242 Visioneers

- Jessica Birks
- David Lubert
- Vijay Oommen
- Gopu Raju
- Shashvat Sinha
- Kevin Zen

*Note to Visioneers* - repo does not contain FEC data because it is too large. It does contain cleaned census data and can also contain polling data.
FEC data might be shared here after sufficient summarization to reduce its size.

## Description

There are four main parts to our code. Installation Data Pulling and Wrangling, Exploratory Data Analysis, Node Graph Processing, Tableau Visualization.

### Data Pulling and Wrangling.

**data_io.py**

Contains code to read in and wrangle raw bulk data from the FEC and to scrape polling data.

Data loosely includes (see docstrings for more specific links): 

* Candidate data
* Committee data
* Individual Contribution Data
* Polling Data (Scraped from Real Clear Politics)

### Exploratory Data Analysis

**plotting.py**

This is a semi-interactive file that was used to create exploratory analysis plots, such as the time series line plots and box and whisker plots, as well as some other plots not shown in the presentation (violin plots).

**cluster.py**

This file contains code to cluster individual contributor occupations to meaningful contribution names. These are clustered using BERT and visualized with TSNE + KMeans. We then classify each occupation using these clusters. 

### Node Graph Processing

**data_processing.py**

This file contains data wrangling code to create the data that feeds the interactive node graph.

### Tableau Visualization: 

WRITE DESCRIPTION HERE.


## Installation For Devs (NO INSTALLATION NECESSARY - Not Needed to Execute)

In terminal build a conda environment and install the dependencies from the environment.yml file.

```
$ make setup
```

## Execution

### Executing the Node Graph

Setup a python server

```
python -m http.server 8000
```

Navigate to the node graph.

```
code -> code -> graph.html
```

