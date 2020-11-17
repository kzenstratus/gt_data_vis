import numpy as np 
import pandas as pd 
import os
# import fec_reader as fec
from typing import List
from plotly.subplots import make_subplots
# from plotly.graph_objs import Bar, Data, Figure, Layout, Marker, Scatter
import plotly.graph_objects as go
import plotly.express as px
from data_processing import get_merged_df, classify_trump_vs_biden


def feature_processing(merged_df, roll_window = "7d"):
    agg_df = merged_df.groupby(["date", "name", "state"]).agg(
                    total_contrib = pd.NamedAgg(column = "transaction_amt", aggfunc = "sum"),
                    contrib_count = pd.NamedAgg(column = "transaction_amt", aggfunc = "count"),
                    poll_point = pd.NamedAgg(column = "value", aggfunc="max") # should only be one value associated with this key.
                ).reset_index()
    agg_df["date"] = pd.to_datetime(agg_df["date"])
    agg_df["contrib_per_person"] = agg_df['total_contrib'] / agg_df['contrib_count']



    # Calc contrib_diff between dates
    agg_df = agg_df.sort_values(["name", "date"]).reset_index(drop = True)
    
    grouped_df = agg_df.groupby("name")
    
    # Moving averages and medians
    for x in ["total_contrib", "contrib_count", "contrib_per_person"]:
        agg_df[f"{roll_window}_{x}_moving_avg"] = grouped_df.rolling(window = roll_window, on = "date")[x].mean().reset_index(drop = True)
        agg_df[f"{roll_window}_{x}_moving_med"] = grouped_df.rolling(window = roll_window, on = "date")[x].median().reset_index(drop = True)
    # Calculate cumulative total contribution.
    agg_df.loc[agg_df["name"] == "Biden", "cumulative_total_contrib_tmp"] = agg_df.loc[agg_df["name"] == "Biden"].total_contrib.cumsum()
    agg_df.loc[agg_df["name"] == "Trump", "cumulative_total_contrib_tmp"] = agg_df.loc[agg_df["name"] == "Trump"].total_contrib.cumsum()
    agg_df.insert(0, value = agg_df["cumulative_total_contrib_tmp"], column = "cumulative_total_contrib")
    agg_df.drop(columns = "cumulative_total_contrib_tmp", inplace = True)
    # grouped_df = agg_df.groupby("name")
    # Calculate other features. Shift here might not work if we are missing days, but we aren't. We'd want to index on date
    # and set freq to D, but not necessary right now.
    agg_df["contrib_diff_7d_avg"] =  grouped_df["7d_total_contrib_moving_avg"].transform(lambda x : x - x.shift(1))
    agg_df["contrib_perc_change_7d_avg"] =  grouped_df["7d_total_contrib_moving_avg"].transform(lambda x : 100 * (x - x.shift(1)) / x.shift(1))
    agg_df["contrib_per_person_diff_7d_avg"] =  grouped_df["7d_contrib_per_person_moving_avg"].transform(lambda x : x - x.shift(1))
    agg_df["contrib_per_person_perc_change_7d_avg"] =  grouped_df["7d_contrib_per_person_moving_avg"].transform(lambda x : 100 * (x - x.shift(1)) / x.shift(1))

    # Moving Avgs for the rest.
    grouped_df = agg_df.groupby("name")

    return agg_df

def make_line_charts(plot_df,
                    cols : List[str] = None,
                    title = "National Polls vs Contributions",
                    out_file = None,
                    height = 2800,
                    width = 1200):

    if cols is None:
        cols = [c for c in plot_df.columns if c not in ["date", "name", "state"]]
    fig = make_subplots(rows = len(cols), cols = 1, subplot_titles = cols)
    biden_df = plot_df.query("name == 'Biden'")
    trump_df = plot_df.query("name == 'Trump'")
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(x=biden_df["date"], y= biden_df[col],
                        mode='lines',
                        name="Biden " + col,
                        line=dict(color = "#002868")),
                        row = i+1, col = 1)
        fig.add_trace(go.Scatter(x=trump_df["date"], y= trump_df[col],
                        mode='lines',
                        name="Trump " + col,
                        line=dict(color = "#BF0A30")),
                        row = i+1, col = 1)

    fig.update_xaxes(matches='x')
    fig.update_layout(height=height
                    , width=width
                    , title_text = title)
    if out_file is not None:
        fig.write_html(out_file)
    return fig

def plot_hist(plot_df,
             cols = ["poll_point"
                    , "total_contrib"
                    , "contrib_count"
                    , "contrib_per_person"
                    , "contrib_diff"
                    , "contrib_perc_change"
                    , "contrib_per_person_diff"
                    , "contrib_per_person_perc_change"
                    ],
                height = 1600,
                width = 1400,
                title = "Contributions by State",
                out_file = None):
    
    fig = make_subplots(rows = len(cols), cols = 1, subplot_titles = cols)
    for i, col in enumerate(cols):
        
        fig.add_trace(go.Violin(x=plot_df['state'][plot_df['name'] == 'Biden'],
                                y=plot_df[col][ plot_df['name'] == 'Biden' ],
                        legendgroup='Biden',
                         scalegroup='Biden',
                          name='Biden',
                        side='negative',
                        line_color='#002868',
                        width = 1
                        ),
                        row = i + 1, col = 1
             )
        fig.add_trace(go.Violin(x=plot_df['state'][ plot_df['name'] == 'Trump' ],
                                y=plot_df[col][ plot_df['name'] == 'Trump' ],
                                legendgroup='Trump',
                                 scalegroup='Trump',
                                  name='Trump',
                                side='positive',
                                line_color="#BF0A30",
                                width = 1),
                                row = i + 1, col = 1)
        # fig = px.violin(plot_df, x = "state", y = "total_contrib", color = "name"
        #                 # , box = True
        #                 )
    fig.update_traces(meanline_visible=True)
    fig.update_layout(height = height,
                        width = width,
                        title_text = title,
                        violingap = 0,
                        violinmode='overlay')
    if out_file is not None:
        fig.write_html(out_file)
    return fig

def plot_occ_violin(merged_df : pd.DataFrame,
                height = 1600,
                width = 1400,
                # title = "National Contribution by Occupation",
                out_dir = None):
    """Quick violin plots per occupation and average donation.

    Args:
        merged_df (pd.DataFrame): contains output of get_merged_df
    """
    # Look at retired people nationally.
    merged_df = merged_df[["state", "date", "transaction_amt", "name", "occupation"]]
    top_10_jobs = list(merged_df.groupby("occupation")["occupation"].count().sort_values(ascending = False)[:10].index)
    merged_df = merged_df.query("occupation.isin(@top_10_jobs)")

    # fig = make_subplots(rows = len(top_10_jobs), cols = 1, subplot_titles = top_10_jobs)
    for i, job in enumerate(top_10_jobs): 
        job_df = merged_df.query("occupation == @job & transaction_amt > 0")
        # Add both sides of the violin plot.
        fig = go.Figure()

        for cand in ["Biden", "Trump"]:
            fig.add_trace(go.Violin(x=job_df['state'][job_df['name'] == cand],
                                y=job_df["transaction_amt"][ job_df['name'] == cand],
                        # legendgroup=cand,
                        #     scalegroup=cand,
                            name=cand,
                        side='negative' if cand == "Biden" else "positive",
                        line_color='#002868' if cand == "Biden" else "#BF0A30",
                        width = 1
                        )
                )
        fig.update_traces(meanline_visible=True)
        fig.data[0].update(span = [job_df.transaction_amt.min(), job_df.transaction_amt.max()], spanmode='manual')
        fig.data[1].update(span = [job_df.transaction_amt.min(), job_df.transaction_amt.max()], spanmode='manual')

        fig.update_layout(height = height,
                            width = width,
                            title_text = f"National Contribution by {job.lower()}",
                            violingap = 0,
                            violinmode='overlay')

                            
        if out_dir is not None:
            fig.write_html(os.path.join(out_dir, f"contrib_by_{job.lower()}.html"))

    return fig


# Do box and whiskers for the following occupations.
# Health Care and Social Assistance
# 0 (Retired/Unemployed),
# 13 (Manufacturing/Construction)
# Educational Services, and 2 (Education).



def plot_occ_box_whisker(merged_df,
                        occupations = ["Retired", "Not-Retired"],
                        occ_col = "label",
                        out_file = "/Users/kevinzen/Data/gt_vis_project/output/moving_avgs/box_whisker_retired.html"):
    plot_df = merged_df[merged_df[occ_col].notnull()][["name", occ_col, "transaction_amt"]]
    
    # plot_df.loc[plot_df["occupation"] != "RETIRED", "occupation"] = "Not-Retired"
    # plot_df.loc[plot_df["occupation"] == "RETIRED", "occupation"] = "Retired"
    # occupations = ["Retired", "Not-Retired"]


    fig = go.Figure()
    for cand in ["Biden", "Trump"]:
        for occ in occupations:
            tmp = plot_df.query(f"name == @cand & {occ_col} == @occ & transaction_amt > 0")["transaction_amt"].describe()
            iqr = tmp["75%"] - tmp["25%"]
            # Extract summary stats because too many data points.
            fig.add_trace(go.Box(y = [np.nan,
                # tmp["min"],
                                      max(tmp["min"], tmp["25%"] - 1.5 * iqr),
                                      tmp["25%"],
                                      tmp["50%"],
                                      tmp["75%"],
                                      min(tmp["max"], tmp["75%"] + 1.5 * iqr),
                                        np.nan
                                    #   tmp["max"]
                                      ]
                                    , name= f"{cand} : {occ}"
                                    , line = dict(color ='#002868' if cand == "Biden" else "#BF0A30"))
                                    )
    fig.update_layout(yaxis = dict(title_text = "Contributions ($)"), 
                            title_text = "National Contributions by Employment")
    if out_file is not None:
        fig.write_html(out_file)
    return fig

def create_all_plots(contrib_df : pd.DataFrame,
                     cmte_df : pd.DataFrame,
                     poll_df : pd.DataFrame,
                     out_dir = "/Users/kevinzen/Data/gt_vis_project/output/") -> None:
    """Create all plots, currently used more as an interactive space where you comment and uncomment
    plots you want to make.

    Args:
        contrib_df (pd.DataFrame): [description]
        cmte_df (pd.DataFrame): [description]
        poll_df (pd.DataFrame): [description]
        out_dir (str, optional): [description]. Defaults to "/Users/kevinzen/Data/gt_vis_project/output/".
    """
    # Create dataframe to feed into timeseries plots.
    # National
    merged_df = get_merged_df(contrib_df = contrib_df
                             , cmte_df = cmte_df
                             , poll_df = poll_df
                             , is_national = True
                             ,more_cols_to_keep = ["cand_name", "label"])
    merged_df = merged_df.query("(date < '2020-10-01') & (transaction_amt > 0)")
    # Plot box and whisker for retired vs non retired.

    # plot_occ_violin(merged_df = merged_df, out_dir = out_dir) # Violin plots for top occupations
    # box_plots = plot_occ_box_whisker(merged_df = merged_df,
    #                                  out_file = os.path.join(out_dir, "box_whisker_retired.html"))
    merged_df["label"] = merged_df["label"].apply(lambda x :  x.capitalize() if type(x) == str else x)
    
    plot_df.loc[plot_df["label"].str.contains("employed|Retired", na  = False), "label"] = "Retired_Unemployed"
    plot_df.loc[plot_df["label"].str.contains("Construction|Manufacturing|Tech|Engineer", na  = False), "label"] = "Engineer_Construction_Trade"
    plot_df.loc[plot_df["label"].str.contains("Education", na  = False), "label"] = "Education"
    plot_df.loc[plot_df["label"].str.contains("Health", na  = False), "label"] = "Healthcare_Medicine"


    for occ in ["Healthcare_Medicine",
                "Engineer_Construction_Trade",
                "Education",
                "Retired_Unemployed"
                ]:
        occ_name = occ.split(" ")[0]
        box_plots = plot_occ_box_whisker(merged_df = plot_df,
                                        occupations = [occ],
                                        out_file = os.path.join(out_dir,
                                            f"box_whisker_{occ_name}.html"))


    plot_df = feature_processing(merged_df = merged_df)
    make_line_charts(plot_df = plot_df,                     
                     out_file = os.path.join(out_dir, "national_poll_contrib.html"))

    # Below currently doesn't work.
    
    # plot_hist(plot_df,
    #             height = 1600,
    #             width = 1400,
    #             title = "National Distribution of Contributions and Polls",
    #             out_file = os.path.join(out_dir, f"national_violin.html"))

    # # By State:
    # merged_df = get_merged_df(contrib_df = contrib_df
    #                          , cmte_df = cmte_df
    #                          , poll_df = poll_df
    #                          , is_national = False)
    # plot_df = feature_processing(merged_df = merged_df)
    # for state in plot_df["state"].unique():
    #     state_df = plot_df.query("state == @state")
    #     make_line_charts(plot_df = state_df
    #                     ,title = f"{state} Polls vs Contribution"
    #                     , out_file = os.path.join(out_dir,f"{state}_poll_contrib.html"))
    
    # # Plot histogram across each state.
    # plot_hist(plot_df,
    #             height = 1600,
    #             width = 1400,
    #             title = "Contributions by State",
    #             out_file = os.path.join(out_dir, f"state_violin.html"))



if __name__ == "__main__":
    cmte_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/cmte_file.csv"
    # cmte_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/committee_candidate_2020.csv"
    # contrib_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/latest_fec_2020_10_30.csv"
    contrib_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/classified_contributions.csv"
    poll_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/polls.csv"
    # Read in data
    cmte_df = pd.read_csv(cmte_file, usecols = ["cmte_id", "name", "cand_name", "party"])

    contrib_df = pd.read_csv(contrib_file,
                             low_memory=True,
                            usecols = ["cmte_id", "state", "transaction_dt", "transaction_amt", "entity_tp", "label"])
    poll_df = pd.read_csv(poll_file, low_memory= False)

    cmte_df = classify_trump_vs_biden(cmte_df = cmte_df)

    create_all_plots(contrib_df = contrib_df,
                     cmte_df = cmte_df,
                     poll_df = poll_df,
                     out_dir = "/Users/kevinzen/Data/gt_vis_project/output/2020_11_08"
                      )    
