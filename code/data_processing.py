import numpy as np 
import pandas as pd 
import os
# import fec_reader as fec
from typing import List
import requests
import json

# fec.get_contributions_to_candidates(start, end)
# reader = fec.DataReader(data_dir='~/Data/gt_vis_project/') # pick a target directory


# fec_file = "~/Data/gt_vis_project/indiv20/by_date/itcont_2020_20200722_20300630.txt"
# fec_file = "~/Data/gt_vis_project/indiv20/by_date/itcont_2020_20200722_20300630.txt"

def get_merged_df(contrib_df,
                 cmte_df,
                 poll_df,
                 more_cols_to_keep = ["cand_name", "occupation"],
                is_national = True,
                merge_type = "outer"):
    # Lets just look at CMTE who donated to trump and biden.
    plot_df = contrib_df.merge(cmte_df.query("cand_name.str.contains('TRUMP|BIDEN')"),
                    on = "cmte_id",
                    how = "inner") # just do inner for now.
    plot_df.loc[plot_df["cand_name"].str.contains("TRUMP"), "cand_name"] = "Trump"
    plot_df.loc[plot_df["cand_name"].str.contains("BIDEN"), "cand_name"] = "Biden"
    plot_df= plot_df[["state", "transaction_dt", "transaction_amt"] + more_cols_to_keep]
    plot_df.rename(columns = {"cand_name" : "name", "transaction_dt": "date"}, inplace = True)
    # Merge in polling data by state.
    if is_national:
        
        plot_df = plot_df.merge(poll_df.query("state == 'National'").drop(columns = ["state"]),
                      on = ["name", "date"],
                      how = merge_type)
        plot_df["state"] = "National"

    else:
        plot_df = plot_df.merge(poll_df.query("state != 'National'"),
                      on = ["name", "date", "state"],
                      how = merge_type)
    return plot_df


def classify_trump_vs_biden(cmte_df: pd.DataFrame) -> pd.DataFrame:
    """Try and classify trump supporting committees vs biden supporting committees.

    Args:
        cmte_df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    trump = "TRUMP, DONALD J."        
    biden = "BIDEN, JOSEPH R JR"
    # All unaffiliated committees which contain the words trump now belong to trump
    # Biden supporters are those against trump.
    cmte_df.loc[(cmte_df["name"].str.contains("TRUMP").fillna(False)) & \
                (cmte_df["party"].isnull()) & \
                (cmte_df["name"].str.contains("AGAINST")), "cand_name"] = biden
    # The rest support trump
    cmte_df.loc[(cmte_df["name"].str.contains("TRUMP").fillna(False)) & \
                (cmte_df["cand_name"].isnull())
                , "cand_name"] = trump

    # Candidates against biden support trump. (Didn't actually find any here)
    cmte_df.loc[(cmte_df["name"].str.contains("BIDEN").fillna(False)) & \
                (cmte_df["party"].isnull()) & \
                (cmte_df["name"].str.contains("AGAINST")), "cand_name"] = trump
    # The rest support trump
    cmte_df.loc[(cmte_df["name"].str.contains("BIDEN").fillna(False)) & \
                (cmte_df["cand_name"].isnull())
                , "cand_name"] = biden
    # Limit to only trump and biden 
    cmte_df = cmte_df.query("cand_name.str.contains('TRUMP|BIDEN').fillna(False)")
    cmte_df.loc[cmte_df["cand_name"].str.contains("BIDEN"), "party"] = "DEM"
    cmte_df.loc[cmte_df["cand_name"].str.contains("TRUMP"), "party"] = "REP"

    cmte_df = cmte_df.drop_duplicates()
    # Some parties contributed to both, ignore the ones that don't make sense.
    cmte_df = cmte_df.query("(cand_name == 'BIDEN, JOSEPH R JR' & party == 'DEM') | (cand_name == 'TRUMP, DONALD J.' & party == 'REP')")
    
    # Some party names still contribute to both parties.
    which_party = pd.pivot(cmte_df.groupby(["name","party"]).count().reset_index(),
                            index = "name",
                            columns = "party",
                            values = "cand_name"
                            ).reset_index()
    which_party = which_party.fillna(0.)
    which_party["true_party"] = which_party.apply(lambda x: "REP" if x.REP > x.DEM else "DEM", axis = 1)
    which_party = which_party[["name", "true_party"]]
    
    cmte_df = cmte_df.merge(which_party, on = "name", how = "left")
    cmte_df["party"] = cmte_df.true_party
    cmte_df.drop(columns = "true_party", inplace = True)
    # Align party with 
    cmte_df.loc[cmte_df["party"] == "DEM", "cand_name"] = "BIDEN, JOSEPH R JR"
    cmte_df.loc[cmte_df["party"] == "REP", "cand_name"] = "TRUMP, DONALD J."
    cmte_df = cmte_df.drop_duplicates()

    return cmte_df

class NodeGraph():
    # Only up til 2020-08-31
    def pre_process(self, merged_df):
        merged_df["date"] = pd.to_datetime(merged_df["date"])
        merged_df["month"] = merged_df["date"].dt.month_name()
        
        # Limit to battleground states.
        battleground = {'AZ' : 11,
                        'MI' : 16,
                        'FL' : 29,
                        'GA' : 16,
                        'MN' : 10,
                        'NC' : 15,
                        'OH' : 18,
                        'PA' : 20,
                        'TX' : 38,
                        'WI' : 10}
        merged_df = merged_df.query("state.isin(@battleground.keys())")
        merged_df["electoral"] = [battleground[state] for state in merged_df["state"]]
        merged_df.rename(columns = {"label" : "industry"}, inplace = True)

        self.merged_df = merged_df
        return merged_df
    
    def get_state_nodes(self, merged_df = None):
        merged_df = self.merged_df if merged_df is None else merged_df
        # State to occupation.

        # Create state nodes.
        keys = ["state",
                #  "month",
                 "name",
                "electoral"]
        agg_df = merged_df.groupby(keys).agg(
                            max_poll = pd.NamedAgg(column = "value", aggfunc = "max")).reset_index()
        static_state_nodes = agg_df[["state", "electoral"]].drop_duplicates()
        static_state_nodes["node_type"] = "state"
        static_state_nodes.rename(columns = {"state" : "node_text"}, inplace = True)
        static_state_nodes["id"] = static_state_nodes["node_text"]
        static_state_nodes.drop_duplicates(inplace = True)
        self.static_state_nodes = static_state_nodes
        
        # Variable Col Names.
        var_state_nodes = pd.pivot(data = agg_df[["state",
                                                #  "month",
                                                  "max_poll", "name"]],
                            index = ["state",
                            # "month"
                            ],
                            columns = "name",
                            values = "max_poll").reset_index()
        # Set colunm names.

        var_state_nodes["name"] = var_state_nodes["state"]
        var_state_nodes.rename(columns = {"state" : "id", "Biden" : "biden_poll", "Trump" : "trump_poll"},
                             inplace = True)
        var_states_nodes = var_state_nodes.drop_duplicates().reset_index(drop = True)
        self.var_state_nodes = var_state_nodes
        return static_state_nodes, var_state_nodes

    def get_state_ind_links(self, merged_df):
        merged_df = reduce_top_10(merged_df, key = ["industry", "state"])
        # Create state <> industry links. Industry Ids are industry + source
        state_links = merged_df[["state", "industry"]].drop_duplicates().reset_index(drop = True)
        state_links["target"] = state_links["industry"] + "_" + state_links["state"]
        state_links.rename(columns = {"state" : "source"}, inplace = True)
        state_links = state_links[["source", "target"]].drop_duplicates().reset_index(drop = True)

        self.state_links = state_links
        return state_links
    
    def get_industry_nodes(self, merged_df):
        merged_df = reduce_top_10(merged_df, key = ["industry", "state"])

        # Static Industry Nodes:

        static_ind_nodes = merged_df[["state", "industry"]].drop_duplicates().reset_index(drop = True)
        static_ind_nodes["id"] = static_ind_nodes["industry"] + "_" + static_ind_nodes["state"]
        static_ind_nodes.rename(columns = {"industry" : "node_text"}, inplace = True)
        static_ind_nodes = static_ind_nodes[["id", "node_text"]].drop_duplicates().reset_index(drop = True)
        static_ind_nodes["node_type"] = "industry"
        static_ind_nodes.drop_duplicates(inplace = True)
        self.static_ind_nodes = static_ind_nodes
        # Variable Industry Nodes ind id, month, transaction sum and count.
        keys = ["state",
                #  "month",
                  "industry"]
        agg_df = merged_df.groupby(keys)\
                          .agg(transaction_sum = pd.NamedAgg(column = "transaction_amt", aggfunc = "sum")
                          ,transaction_count = pd.NamedAgg(column = "transaction_amt", aggfunc = "count")
                          ).reset_index()
        agg_df["avg_contrib"] = agg_df["transaction_sum"] / agg_df["transaction_count"]
        agg_df["id"] = agg_df["industry"] + "_" + agg_df["state"]
        var_ind_nodes = agg_df[["id",
                            #  "month",
                              "transaction_sum", "transaction_count", "avg_contrib"]]\
                        .drop_duplicates().reset_index(drop = True)
        
        self.var_ind_nodes = var_ind_nodes
        return static_ind_nodes, var_ind_nodes

    def get_ind_cmte_links(self, merged_df):

        merged_df = reduce_top_10(merged_df, key = ["state", "industry"])
        merged_df = reduce_top_10(merged_df, key = ["state", "cmte_name"])

        ind_links = merged_df[["state", "industry", "cmte_name"]].drop_duplicates().reset_index(drop = True)
        ind_links["source"] = ind_links["industry"] + "_" + ind_links["state"]
        ind_links.rename(columns = {"cmte_name" : "target"}, inplace = True)
        ind_links = ind_links[["source", "target"]].drop_duplicates().reset_index(drop = True)
        
        self.ind_links = ind_links
        return ind_links
    def get_cmte_nodes(self, merged_df):
        merged_df = reduce_top_10(merged_df, key = ["state", "cmte_name"])

        # Static Committee Nodes:
        static_cmte_nodes = merged_df[["cmte_name", "party"]].drop_duplicates().reset_index(drop = True)
        static_cmte_nodes["id"] = static_cmte_nodes["cmte_name"]
        static_cmte_nodes["node_type"] = "committee"
        static_cmte_nodes.rename(columns = {"cmte_name" : "node_text"}, inplace = True)
        static_cmte_nodes = static_cmte_nodes.query("node_text.notnull()")
        # static_cmte_nodes.loc[static_cmte_nodes["party"] == "Democrat", "party"] = "Dem"
        # static_cmte_nodes.loc[static_cmte_nodes["party"] == "Republican", "party"] = "Rep"
        self.static_cmte_nodes = static_cmte_nodes

        # Variable committee nodes.

        keys = [
            # "month",
         "cmte_name"]
        var_cmte_nodes = merged_df.groupby(keys)\
                          .agg(transaction_sum = pd.NamedAgg(column = "transaction_amt", aggfunc = "sum")
                          ,transaction_count = pd.NamedAgg(column = "transaction_amt", aggfunc = "count")
                          ).reset_index()
        var_cmte_nodes["avg_contrib"] = var_cmte_nodes["transaction_sum"] / var_cmte_nodes["transaction_count"]
        var_cmte_nodes.rename(columns = {"cmte_name" : "id"}, inplace = True)
         
        self.var_cmte_nodes = var_cmte_nodes

        return static_cmte_nodes, var_cmte_nodes

    def get_files_for_node_graph(self, merged_df, out_dir = None):
        merged_df = reduce_top_10(merged_df, key = ["state", "cmte_name"])


        # Create each node and link.
        merged_df = self.pre_process(merged_df)
        static_state_nodes, var_state_nodes = self.get_state_nodes(merged_df)
        print(f"Static state nodes : {static_state_nodes.shape}, var : {var_state_nodes.shape}")

        state_links = self.get_state_ind_links(merged_df)
        print(f"State Links : {state_links.shape}")

        static_ind_nodes, var_ind_nodes = self.get_industry_nodes(merged_df)
        print(f"static ind node : {static_ind_nodes.shape}, var {var_ind_nodes.shape}")

        ind_links = self.get_ind_cmte_links(merged_df)
        print(f"Ind Links : {state_links.shape}")

        static_cmte_nodes, var_cmte_nodes = self.get_cmte_nodes(merged_df)
        print(f"static cmte node : {static_cmte_nodes.shape}, var {var_cmte_nodes.shape}")

        # Make final 3 Tables.
        links = state_links.append(ind_links)
        links = links.query("source.notnull() & target.notnull()").drop_duplicates()
        print(f"links : {links.shape}")

        static_nodes = static_state_nodes.append([static_ind_nodes, static_cmte_nodes])
        print(f"static nodes: {static_nodes.shape}")

        var_nodes = var_state_nodes.append([var_ind_nodes, var_cmte_nodes])
        print(f"var nodes: {var_nodes.shape}")


        return links, static_nodes, var_nodes

def reduce_top_10(merged_df, key = ["industry", "state"]):
    tmp_df = merged_df.groupby(key)\
                        .agg(total_amt = pd.NamedAgg("transaction_amt", "sum"))\
                        .reset_index()
    tmp_df = tmp_df.sort_values(["state", "total_amt"], ascending = False)
    ind_to_keep = tmp_df.merge(tmp_df.groupby("state")["total_amt"].head(10))

    tmp_df = merged_df.merge(ind_to_keep[key], on = key, how = "inner")
    
    return tmp_df

def clean_cmte_df(cmte_df):
    cmte_df = classify_trump_vs_biden(cmte_df = cmte_df)

    cmte_df.rename(columns = {"name" : "cmte_name"},
                     inplace = True)

    cmte_df = cmte_df.query("cmte_name.notnull()")
    # Normalize Committee names Lots of BIDEN FOR PRESIDENT
    cmte_df.loc[cmte_df.cmte_name.str.contains("BIDEN FOR PRESIDENT"), "cmte_name"] = "BIDEN FOR PRESIDENT"
    cmte_df.loc[cmte_df.cmte_name.str.contains("TRUMP FOR PRESIDENT"), "cmte_name"] = "DONALD J. TRUMP FOR PRESIDENT"
    
    return cmte_df

if __name__ == "__main__":
    cmte_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/cmte_file.csv"
    # cmte_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/committee_candidate_2020.csv"
    # contrib_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/latest_fec_2020_10_30.csv"
    contrib_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/classified_contributions.csv"
    poll_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/polls.csv"
    out_dir = "/Users/kevinzen/Data/gt_vis_project/node_graph"
    # Read in data
    cmte_df = pd.read_csv(cmte_file, usecols = ["cmte_id", "name", "cand_name", "party"])
    cmte_df = clean_cmte_df(cmte_df)
    contrib_df = pd.read_csv(contrib_file,
                             low_memory=True,
                            usecols = ["cmte_id", "state", "transaction_dt", "transaction_amt", "entity_tp", "label"])
    
    # Limit contributions to top 10 labels in each state.
    # 203883540 for biden C00703975
    # 82542975 for trump. C00580100
    poll_df = pd.read_csv(poll_file, low_memory= False)



    merged_df = get_merged_df(contrib_df = contrib_df
                            , cmte_df = cmte_df
                            , poll_df = poll_df
                            , is_national = False
                            , more_cols_to_keep = ["cand_name", "label", "cmte_id", "cmte_name", "party"]
                            )
    merged_df = merged_df.query("(date <= '2020-10-15') & (transaction_amt > 0)")

    node_data = NodeGraph()
    links, static_nodes, var_nodes = node_data.get_files_for_node_graph(merged_df)
    links.to_csv(os.path.join(out_dir, "small_links_all_dates.csv"))
    static_nodes.to_csv(os.path.join(out_dir, "small_nodes_all_dates.csv"))
    var_nodes.to_csv(os.path.join(out_dir, "small_var_nodes_all_dates.csv"))
    # Some committees still contributed to both, take the majority.


# if __name__ == "__main__":
#     cmte_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/cmte_file.csv"
#     # cmte_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/committee_candidate_2020.csv"
#     # contrib_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/latest_fec_2020_10_30.csv"
#     contrib_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/classified_contributions.csv"
#     poll_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/polls.csv"
#     # Read in data
#     cmte_df = pd.read_csv(cmte_file, usecols = ["cmte_id", "name", "cand_name", "party"],
#                             dtype = {"cand_name" : str})
#     cmte_df["cand_name"] = cmte_df["cand_name"].astype(str)
#     cmte_df = cmte_df.query("cand_name.str.contains('TRUMP|BIDEN')")
#     cmte_df = cmte_df.drop_duplicates()
#     cmte_df.rename(columns = {"name" : "cmte_name"}, inplace = True)
#     # Some parties contributed to both, ignore the ones that don't make sense.
#     cmte_df = cmte_df.query("(cand_name == 'BIDEN, JOSEPH R JR' & party == 'DEM') | (cand_name == 'TRUMP, DONALD J.' & party == 'REP')")
#     contrib_df = pd.read_csv(contrib_file,
#                              low_memory=True,
#                             usecols = ["cmte_id", "state", "transaction_dt", "transaction_amt", "entity_tp", "label"])
#     poll_df = pd.read_csv(poll_file, low_memory= False)



    # create_all_plots(contrib_df = contrib_df,
    #                  cmte_df = cmte_df,
    #                   poll_df = poll_df,
    #                 out_dir = "/Users/kevinzen/Data/gt_vis_project/output/moving_avgs"

    #                   )    
    

