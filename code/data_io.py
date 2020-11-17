import numpy as np 
import pandas as pd 
import os
# import fec_reader as fec
from typing import List
import requests
import json

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

def get_cmte_df(cmte_file,
                 cmte_header,
                 cand_file,
                 cmte_summary_file, 
                 out_file = "/Users/kevinzen/Data/gt_vis_project/clean_data/cmte_file.csv"):
    """Create a committee data frame that shows who committees donated to and their 
    political leanings (republican vs democrat).

    Args:
        cmte_file ([str]): itpas2.txt from https://www.fec.gov/files/bulk-downloads/2020/pas220.zip
        cmte_header ([str]): pas2_header_file.csv from https://www.fec.gov/files/bulk-downloads/data_dictionaries/pas2_header_file.csv
        cand_file ([str]): All candidates.csv from https://www.fec.gov/campaign-finance-data/all-candidates-file-description/
        cmte_summary_file ([str]): committee_summary_2020.csv from https://www.fec.gov/files/bulk-downloads/2020/candidate_summary_2020.csv
    Example:

    # cmte summary file here - https://www.fec.gov/data/browse-data/?tab=committees
    # somehow mater file doesn't have all committees. ie. make america great again . C00618371
    # is a joint committee.
    >>> cmte_file = "/Users/kevinzen/Data/gt_vis_project/raw/itpas2.txt"
    >>> cmte_header = "/Users/kevinzen/Data/gt_vis_project/raw/pas2_header_file.csv"
    >>> cand_file = "/Users/kevinzen/Data/gt_vis_project/raw/All candidates.csv"
    >>> cmte_summary_file = "/Users/kevinzen/Data/gt_vis_project/raw/committee_summary_2020.csv"
    >>> cmte_df = get_cmte_df(cmte_file, cmte_header, cand_file, cmte_summary_file)
    """
    cmte_df = pd.read_csv(cmte_file,
                         low_memory = False,
                         delimiter= '|',
                         header= None
                          )
    col_names = pd.read_csv(cmte_header)
    col_names = list(col_names.columns)
    cmte_df.columns = [x.lower() for x in col_names]
    
    # Ignore non committees.
    cmte_df = cmte_df.query("~entity_tp.isin(['ORG', 'IND'])")

    cand_df = pd.read_csv(cand_file
        , usecols = ["CAND_ID", "CAND_NAME", "CAND_PTY_AFFILIATION"]
        )
    cand_df.columns = ["cand_id", "cand_name", "party"]
    cmte_df = cmte_df[["cmte_id", "name", "transaction_dt", "transaction_amt", "cand_id"]]
    cmte_df = cmte_df.merge(cand_df, on = "cand_id")
    # Aggregate all contributions given from a committee to a gien candidate and party.
    cmte_df = cmte_df[cmte_df["party"].isin(["DEM", "REP"])]
    # Find which committee is associated to each party.
    which_party = cmte_df.groupby(["party", "cmte_id"])["transaction_amt"].sum().reset_index()
    which_party = which_party.groupby('cmte_id', group_keys=False).apply(lambda x: x.loc[x.transaction_amt.idxmax()]).reset_index(drop=True)
    
    # Map each committee to their given party, but keep the candidate name and id.
    # Probably could use transform for this.
    cmte_df.drop(columns = ["party"], inplace = True)
    cmte_df = cmte_df.merge(which_party[["cmte_id", "party"]], on = "cmte_id", how = "left")
    # Test on C00530766, Biden gets thrown in as a Rep.
    # cmte_df.query("cmte_id == 'C00530766' & cand_name.str.contains('BIDEN')")
    
    # Read in all committee files. Not all committees show up in the committee contribution file.
    # Infer from the name what these committees are.
    cmte_m_df = pd.read_csv(cmte_summary_file)
    cmte_m_df.columns = [x.lower() for x in list(cmte_m_df.columns)]
    cmte_m_df = cmte_m_df[["cmte_id", "cmte_nm", "cmte_dsgn"]]
    cmte_m_df.rename(columns = {"cmte_nm" : "name"}, inplace = True)
    
    # No need to join by candidate id because committee contribution should have us covered.
    # Committees who didn't make contributions won't have a cand name anyways.
    
    # Join in committees since not all committees are in the committee contribution file.
    cmte_df = cmte_m_df.merge(cmte_df, on = ["cmte_id", "name"], how = "outer")

    cmte_df.to_csv(out_file, index = False)
    return cmte_df

def poll_to_df(raw_poll_json : dict) -> pd.DataFrame:
    """Helper function convert polling data to data frame.

    Args:
        raw_poll_json (dict): JSON scraped from Real clear politics.

    Returns:
        pd.DataFrame: polling dataframe
    """
    rcp_avgs = raw_poll_json['poll']['rcp_avg']
    final_df = pd.DataFrame()
    for i, val in enumerate(rcp_avgs):
        tmp = pd.json_normalize(val['candidate'])
        tmp['date'] = val['date']
        final_df = final_df.append(tmp)
    final_df.drop(columns = ['id', 'status', 'color'], inplace = True)
    return final_df

def pull_latest_rcp(out_file : str = "~/Data/gt_vis_project/clean_data/polls.csv"):
    """Pull latest Real Clear Politics Data by state. and national.
    https://onlinejournalismblog.com/2017/05/10/how-to-find-data-behind-chart-map-using-inspector/
    Args:
        out_file (str, optional): [description]. Defaults to "~/Data/gt_vis_project/clean_data/polls.csv".

    Returns:
        [type]: [description]
    Example:
    >>> pull_latest_rcp()
    """
# 

    rcp_jsons = {"North Carolina" : "https://www.realclearpolitics.com/epolls/json/6744_historical.js?1602635842295&callback=return_json",
                 "Wisconsin" : "https://www.realclearpolitics.com/epolls/json/6849_historical.js?1602638476165&callback=return_json",
                 "Florida" : "https://www.realclearpolitics.com/epolls/json/6841_historical.js?1602638538981&callback=return_json",
                 "Michigan": "https://www.realclearpolitics.com/epolls/json/6761_historical.js?1602638561657&callback=return_json",
                 "Pennsylvania" : "https://www.realclearpolitics.com/epolls/json/6861_historical.js?1602638584280&callback=return_json",
                 "Arizona" : "https://www.realclearpolitics.com/epolls/json/6807_historical.js?1602638600473&callback=return_json",
                "Ohio": "https://www.realclearpolitics.com/epolls/json/6765_historical.js?1602638729359&callback=return_json",
                "Minnesota" : "https://www.realclearpolitics.com/epolls/json/6966_historical.js?1602638770015&callback=return_json",
                "Iowa": "https://www.realclearpolitics.com/epolls/json/6787_historical.js?1602638787908&callback=return_json",
                "Texas" : "https://www.realclearpolitics.com/epolls/json/6818_historical.js?1602638819149&callback=return_json",
                "Georgia" : "https://www.realclearpolitics.com/epolls/json/6974_historical.js?1602638840551&callback=return_json",
                # Nothing from Virgina or Nevada or Colorado, or new mexico.
                "New Hampshire" : "https://www.realclearpolitics.com/epolls/json/6779_historical.js?1602638879306&callback=return_json",
                "Maine" : "https://www.realclearpolitics.com/epolls/json/6922_historical.js?1602638900859&callback=return_json",
                 "National" :"https://www.realclearpolitics.com/epolls/json/6247_historical.js?1602638622255&callback=return_json"
                 }
    final_df = pd.DataFrame()
    for state, url in rcp_jsons.items():
        print(state)
        # Pull in raw json
        rv = requests.get(url)
        rv = rv.content.decode('utf-8') # byte to string
        # 12:2 is to remove the return_json() characters.
        print("converting to dict")
        raw_poll_json = json.loads(rv[12:-2]) # string to dict.
            
        # Convert to df
        print("converting to df")
        tmp_df = poll_to_df(raw_poll_json)
        tmp_df['state'] = state
        final_df = final_df.append(tmp_df)
    final_df["state"] = [us_state_abbrev[state] if state in us_state_abbrev.keys() else state for state in final_df["state"]]
    final_df["date"] = pd.to_datetime(final_df["date"], utc = True).dt.date
    final_df.to_csv(out_file, index = False)

    return final_df

def ingest_single_fec(contrib_file : str,
                      contrib_header_file : str,
                      min_date = "2020-04-01") -> pd.DataFrame:
    """
    Read in fec contrib data from raw bulk download.
    Args:
        contrib_file (str): [description]
        contrib_header_file (str): [description]
        min_date (str, optional): [description]. Defaults to "2020-04-01".

    Returns:
        pd.DataFrame: [description]
    
    Example:
    >>> contrib_file = 
    """
    print(contrib_file)
    fec_df = pd.read_csv(contrib_file,
                         low_memory = False,
                         delimiter= '|',
                         header= None
                        #   error_bad_lines= False
                          )
    col_names = pd.read_csv(contrib_header_file)
    col_names = list(col_names.columns)
    
    fec_df.columns = [x.lower() for x in col_names]
    fec_df['transaction_dt'] = pd.to_datetime(fec_df['transaction_dt'], format="%m%d%Y")
    fec_df.drop(columns = ["image_num", "sub_id", "memo_cd", "file_num", "tran_id"], inplace = True)

    fec_df = fec_df[fec_df['transaction_dt'] > pd.to_datetime(min_date)]
    return fec_df

def read_all_fec_data(fec_dir : str,
                      contrib_header_file : str,
                      out_file : str) -> pd.DataFrame:
    """Read in and aggregate all FEC contrib data into one file, removing uncessary data.

    Args:
        fec_dir (str): the by_date directory of the fec indiv download.
        out_file (str): absolute path to the aggregated output file.

    Returns:
        pd.DataFrame: Starting from april 2020 onwords.
    Examples:
    >>> fec_dir = "/Users/kevinzen/Data/gt_vis_project/indiv20/by_date/"
    >>> contrib_header_file = "/Users/kevinzen/Data/gt_vis_project/indiv_header_file.csv"
    >>> all_df = read_all_fec_data(fec_dir = fec_dir,
                contrib_header_file = contrib_header_file,
             out_file = os.path.join(fec_dir, "latest_fec.csv"))
    """
    all_files = os.listdir(fec_dir)
    all_files = [x for x in all_files if ("invalid" not in x and "2019" not in x)]
    all_df = pd.concat([ingest_single_fec(contrib_file = os.path.join(fec_dir, x),
                                         contrib_header_file = contrib_header_file) for x in all_files],
          ignore_index=True)
    all_df.to_csv(out_file, index = False)
    return all_df

