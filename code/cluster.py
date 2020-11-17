import pandas as pd
import numpy as np
from typing import List, Dict
import re
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go


def make_tsne_plots(contrib_df, out_file):
    """Make TSNE Plots for top 100 occupations.

    Args:
        contrib_file ([type]): [description]
        out_file ([type]): [description]
    """
    corpus = contrib_df.groupby("occupation").agg(count = pd.NamedAgg("occupation", "count")).reset_index()
    corpus.sort_values("count", ascending = False, inplace = True)
    corpus = corpus.query("occupation.notnull()")
    
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    small_corpus = list(corpus[:1000]["occupation"])
    corpus_embeddings = embedder.encode(small_corpus)
    
    num_clusters = 15
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)

    cluster_assignment = clustering_model.labels_
    # clustered_sentences = [[] for i in range(num_clusters)]
    
    # Reduce Dimensions for visualization
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(corpus_embeddings)
    
    plot_df = pd.DataFrame()
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        tmp_df = pd.DataFrame({"sentence_id" : sentence_id,
                               "cluster_id" : cluster_id,
                               "name" : small_corpus[sentence_id],
                               "x" : tsne_results[sentence_id][0],
                               "y" : tsne_results[sentence_id][1]
                              }, index = [0])
        plot_df = plot_df.append(tmp_df)

    plot_df["cluster_id"] = plot_df.cluster_id.astype(str)
    # visualize embedding
    fig = px.scatter(plot_df, x = "x", y = "y", color = "cluster_id", hover_data = ["name"])
    
    fig.update_layout(title_text = "TSNE - Top 100 Occupations Clustered (K Means) ")
    if out_file is not None:
        fig.write_html(out_file)

    return fig

def calc_cos_sim(vec_a, vec_b):
    # norm is the size of each vector, ie. [1,0] has size 1, and [1,1] has sqrt(2) norm.
    # Angle between two vectors can be represented as a dot product.
    # Cosine similarity isn't the degree between the two vector, rather it's between [-1,1]
    # where 90 deg -> 0, 180 -> -1 similarity.
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def _expand_naics_range_helper(code_range : str) -> List[str]:
    """Tries to expand strings of num1-num2, eg. (40-45) to [40,41, 42, 43, 44, 45]
    Args:
        code_range (str): [description]

    Returns:
        List[str]: [description]
    """
    if "-" in code_range:
        try:
            start, end = re.match("([0-9]*?)-([0-9]*)", code_range).group(1,2)
            start, end = int(start), int(end)
            return([str(x) for x in range(start, end + 1)])
        except Exception as err:
            print(err)
    return([code_range])

def expand_naics_range(code_col: List[str], val_col: List[str]):
    rv = pd.DataFrame()
    for i, code in enumerate(code_col):
        if "-" in code:
            tmp_codes = pd.DataFrame(dict(code = _expand_naics_range_helper(code_range = code)))
            tmp_codes["title"] = val_col[i]
            rv = rv.append(tmp_codes)
    return rv


def get_naics_codes(naics_map_url : str = "http://api.naics.us/v0/q?year=2012",
                    limit_2_dig = False) -> pd.DataFrame:
    """Get two digit naics codes to use as industry labels for people's occupations.
    Args:
        naics_map_url (str, optional): url or file where naics codes are stored. Defaults to "http://api.naics.us/v0/q?year=2012".

    Returns:
        pd.DataFrame: 
    """
    naics_df = pd.read_json(naics_map_url)
    # restrict to two digits.
    naics_df["code"] = naics_df["code"].astype(str)
    if limit_2_dig is False:
        return naics_df
    naics_df_final = naics_df[naics_df["code"].apply(lambda x : len(x) < 3 or "-" in x)]

    # Lets concatenate all full naics codes under a 2 digit naics code.
    range_df = expand_naics_range(code_col = list(naics_df_final.code), val_col = list(naics_df_final.title))
    range_df = range_df.merge(naics_df_final.drop("code", inplace = False, axis = 1), on = "title")
    naics_df_final = naics_df_final.append(range_df)
    naics_df_final = naics_df_final.query("~code.str.contains('-')")


    # Lets concatenate all titles 
    full_titles_df = naics_df[naics_df["code"].apply(lambda x : len(x) >= 3 and "-" not in x)]
    # Convert full naics codes to 
    full_titles_df["code"] = full_titles_df["code"].apply(lambda x : x[:2])
    full_titles_df = full_titles_df.groupby("code")["title"].agg(all_titles = lambda x : ", ".join(x)).reset_index()

    # Merge in constructed titles into the rest of the 2 digit naics codes.

    naics_df_final = full_titles_df.merge(naics_df_final[["code", "title", "description"]], on = "code", how = "left")
    return naics_df_final
    

def classify_text(input_embeddings,
                input_df,
                label_text ,
                label_text_2 = None, 
                input_uid_col = "occupation",
                embedder = None):
    """[summary]

    Args:
        input_embeddings ([type]): [description]
        input_df ([type]): [description]
        label_text ([type]): [description]
        label_text_2 ([type], optional): Embed label text but use this as the actual label. Defaults to None.
        input_uid_col (str, optional): [description]. Defaults to "occupation".
        embedder ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if len(input_embeddings) != input_df.shape[0]:
        print("ERROR - embedding length not equal to input df length.")
        return

    if embedder is None:
        embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    label_embedding = embedder.encode(label_text)

    # For each observed sentence, find the closest label.
    best_label = []
    highest_val = []

    print(f"Starting to classify input_df : {input_df.shape}, label_text : {len(label_text)}")
    if label_text_2 is not None:
        label_text = label_text_2
    for i, obs_val in enumerate(corpus[input_uid_col].values):
        if i % 10000 == 0:
            print(f"iteration {i}, {obs_val}")
        obs_vec = input_embeddings[i]
        top_label = (None, -2)
        for j, label in enumerate(label_text):
            label_vec = label_embedding[j]
            sim = calc_cos_sim(obs_vec, label_vec)
            if sim > top_label[1]:
                top_label = (label, sim)
        
        best_label.append(top_label[0])
        highest_val.append(top_label[1])
    
    input_df["label"] = best_label
    input_df["label_val"] = highest_val

    return input_df


        
def classify_industry(corpus_embeddings, corpus, embedder = None):
    labels_df = get_naics_codes()
    # We can try this on both the title and the description.
    # Convert description to a string.
    labels_df["description"] = labels_df["description"].apply(lambda x : "".join(x))
    rv_df = classify_text(input_embeddings = corpus_embeddings,
                 input_df = corpus,
                 label_text = labels_df["all_titles"].values,
                 label_text_2 = labels_df["title"].values,
                 input_uid_col = "occupation",
                 embedder = embedder)
    # using only title, the similarity is as follows.
    # mean          0.659431
    # std           0.088902
    # min           0.198436
    # 25%           0.607540
    # 50%           0.668285
    # 75%           0.718704
    # max           1.000000
    
    # using description
    # mean          0.479111
    # std           0.094440
    # min           0.122354
    # 25%           0.417947
    # 50%           0.480547
    # 75%           0.542711
    # max           0.793314

    # Use all naics codes within a 2 digit naics code.
    # Things like attorney still get classified incorrectly with the above methods.
    # mean          0.503932
    # std           0.097924
    # min           0.152747
    # 25%           0.441259
    # 50%           0.507282
    # 75%           0.570093
    # max           0.823853
    
    # Or we can just use all industries...

    # For labels that are are below 20% similarity and show up more than 1000 times will get manually 
    # labeled.
    # rv_df.query("label_val < 0.30 & count > 1000" ) ~ 190.
    rv_df.loc[(rv_df["label_val"] < 0.30) & (rv_df["count"] > 1000), "label"] = rv_df["occupation"]
    return rv_df    

if __name__ == "__main__":
    # occupation file
    contrib_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/latest_fec_2020_10_30.csv"
    out_file = "/Users/kevinzen/Data/gt_vis_project/output/tsne_100_occupation_2020_10_30.html"
    contrib_df = pd.read_csv(contrib_file,
                            low_memory=True,
                            usecols = ["cmte_id",
                                         "state",
                                         "transaction_dt",
                                         "transaction_amt",
                                         "entity_tp",
                                         "occupation"]
                                         )

    corpus = contrib_df.groupby("occupation").agg(count = pd.NamedAgg("occupation", "count"),
                                                  total_transaction = pd.NamedAgg("transaction_amt", "sum")).reset_index()
    corpus.sort_values(["count", "total_transaction"], ascending = False, inplace = True)
    corpus = corpus.query("occupation.notnull() & total_transaction > 0")
    

    # corpus.total_transaction.hist()
    # fig = px.histogram(corpus.query("total_transaction < 1000"), x = "total_transaction")
    # fig.show()    

    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    # small_corpus = list(corpus[:1000]["occupation"])
    corpus_embeddings = embedder.encode(corpus.occupation.unique(), show_progress_bar=True, num_workers=8)
    np.save('/Users/kevinzen/Data/gt_vis_project/embedding/all_occ_embeddings.npy', np.array(corpus_embeddings))
    # Embedding the above takes ~ 20 min, load in the pre-embedded vals.
    corpus.to_csv('/Users/kevinzen/Data/gt_vis_project/embedding/embedding_corpus.csv', index = False)

    classified_df = classify_industry(corpus_embeddings, corpus, embedder = None)
    classified_df.to_csv('/Users/kevinzen/Data/gt_vis_project/embedding/agg_classified_occupations.csv', index = False)
    # Lets map this back to the committees.
    classified_contrib_df = classified_df[["occupation", "label", "label_val"]].merge(contrib_df, on = "occupation", how = "right")
    classified_contrib_df.to_csv("/Users/kevinzen/Data/gt_vis_project/clean_data/classified_contributions.csv", index = False)

    fig = make_tsne_plots(contrib_df, out_file = out_file)


    # fig = go.Figure()

    # fig.add_trace(
    #     go.Scattergl(
    #         x = plot_df["x"],
    #         y = plot_df["y"],
    #         mode = 'markers',
    #         marker = dict(
    #             color = "#002868",
    #             line = dict(
    #                 width = 1,
    #                 color = 'DarkSlateGrey')
    #         )
    #     )
    # )

    # fig.show()


    pass