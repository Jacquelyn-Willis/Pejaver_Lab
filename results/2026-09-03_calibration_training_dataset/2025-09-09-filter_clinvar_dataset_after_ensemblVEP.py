import sys
import pandas as pd 
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import requests
import tarfile
#!{sys.executable} -m pip install requests

#directories 
mount_data = "/Users/jwillis/minerva/pejaverlab/data/2026-09-03_calibration_training_dataset"
mount_results = "/Users/jwillis/minerva/pejaverlab/results/2026-09-03_calibration_training_dataset"

data= "/sc/arion/projects/pejaverlab/users/willij115/data/2026-09-03_calibration_training_dataset" 
results= "/sc/arion/projects/pejaverlab/users/willij115/results/2026-09-03_calibration_training_dataset"


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



### 1. upload ensembl VEP output files for hg38

def read_vep_output(vep_file_path):
    

    # Find the line containing the actual column names
    with open(vep_file_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith("#Uploaded_variation"):
            header_line = i
            columns = line[1:].rstrip("\n").split("\t")
            break


    # Read everything after the VEP header
    vep_df = pd.read_csv(
        vep_file_path,
        sep="\t",
        skiprows=header_line + 1,
        names=columns,
        low_memory=False
    )
    
    return vep_df






#### 2a. filter for mane select transcripts for hg38

def filter_mane_select_transcripts(vep_df):
    mane_select_transcripts = vep_df[
        vep_df["MANE"] == "MANE_Select"
    ].copy()
    
    return mane_select_transcripts



##### 2b. filter for canonical transcripts for hg37

def filter_canonical_transcripts(vep_df):
    canonical_transcripts = vep_df[
        vep_df["CANONICAL"] == "YES"
    ].copy()
    
    return canonical_transcripts




#### 3. filter vep output for missense variants 

def filter_missense_variants(vep_df):
    vep_missense = vep_df[
        vep_df["Consequence"] == "missense_variant"
    ].copy()
    
    return vep_missense





#### 4. filter for AF_gnomade and AF_gnomadg for hg38 < 0.01

def filter_AF_gnomad(missense_filtered_df):
   
    df = missense_filtered_df.copy()
    # Use gnomAD exome AF when available;
    # otherwise use gnomAD genome AF
    df["AF"] = (
        df["gnomADe_AF"]
        .combine_first(df["gnomADg_AF"])
    )

    df["AF"] = pd.to_numeric(
        df["AF"],
        errors="coerce"
    )


    AF_filtered_df = df[
        df["AF"] < 0.01
    ].copy()
    
    return AF_filtered_df





### Function calls


vep_out_hg38_df = read_vep_output(os.path.join(mount_results,
    "clinvar_filtered_for_ensemblVEP_hg38_vep_output.txt"))

vep_out_hg37_df = read_vep_output(os.path.join(mount_results,
    "clinvar_filtered_for_ensemblVEP_hg37_vep_output.txt"))
    

mane_select_transcripts_hg38 = filter_mane_select_transcripts(vep_out_hg38_df)
canonical_transcripts_hg37 = filter_canonical_transcripts(vep_out_hg37_df)

    
vep_missense_hg38 = filter_missense_variants(mane_select_transcripts_hg38)
vep_missense_hg37 = filter_missense_variants(canonical_transcripts_hg37)

AF_filtered_hg38 = filter_AF_gnomad(vep_missense_hg38)  
AF_filtered_hg37 = filter_AF_gnomad(vep_missense_hg37)  


final_hg38 = AF_filtered_hg38.copy()
final_hg37 = AF_filtered_hg37.copy()


final_hg38.to_csv(
        os.path.join(mount_results, "clinvar_hg38_post_ensemble_vep_filters_no_uniprot.tsv"),
        sep="\t",
        index=False
    )

final_hg37.to_csv(
        os.path.join(mount_results, "clinvar_hg37_post_ensemble_vep_filters_no_uniprot.tsv"),
        sep="\t",
        index=False
    )



final_hg38.to_csv(
        os.path.join(mount_results, "clinvar_hg38_post_ensemble_vep_filters_w_uniprot.tsv"),
        sep="\t",
        index=False
    )

final_hg37.to_csv(
        os.path.join(mount_results, "clinvar_hg37_post_ensemble_vep_filters_w_uniprot.tsv"),
        sep="\t",
        index=False
    )

























'''
step_dataframes = {

            "vep_out_hg38_df": vep_out_hg38_df,
            "vep_out_hg37_df": vep_out_hg37_df,
            "MANE_Select_hg38": mane_select_transcripts_hg38,
            "Canonical_hg37": canonical_transcripts_hg37,
            "missense_hg38": vep_missense_hg38,
            "missense_hg37": vep_missense_hg37,
            "AF_hg38": AF_filtered_hg38,
            "AF_hg37": AF_filtered_hg37,
        
}

df_summary = build_filter_summary_table(step_dataframes)

display(df_summary)

plot_filter_summary(df_summary, metric="UniqueVariationID")
plot_filter_summary(df_summary, metric="UniqueGeneSymbol")
'''


#remove mutpred2 training variants that overlap 


    

#REmove polyphen2 training variants also
def load_poly_phen_data ():
    
     
    tar_path = os.path.join(
        mount_data,
        "training-2.2.2.tar.gz"
    )

    polyphen_files = [
        "humdiv-2011_12.deleterious.pph.input",
        "humdiv-2011_12.neutral.pph.input",
        "humvar-2011_12.deleterious.pph.input",
        "humvar-2011_12.neutral.pph.input",
    ]

    polyphen_dfs = []

    with tarfile.open(tar_path, "r:gz") as tar:

        for filename in polyphen_files:

            member = tar.getmember(filename)

            with tar.extractfile(member) as f:

                df = pd.read_csv(
                    f,
                    sep="\t",
                    header=None,
                    names=[
                        "uniprot_id",
                        "position",
                        "ref_aa",
                        "alt_aa"
                    ]
                )

                polyphen_dfs.append(df)

    # Combine all four files
    polyphen_train = pd.concat(
        polyphen_dfs,
        ignore_index=True
    )

    print(polyphen_train.head())
    print(polyphen_train.shape)
    
    return polyphen_train

poly_phen_training_set = load_poly_phen_data()





def get_uniprot_ids_batch(transcripts, genome="hg38", batch_size=100):
    
    if genome == "hg38":
        base_url = "https://rest.ensembl.org"
    elif genome == "hg37":
        base_url = "https://grch37.rest.ensembl.org"
    else:
        raise ValueError("genome must be 'hg38' or 'hg37'")

    url = f"{base_url}/xrefs/id"

    results_map = {}

    for start in range(0, len(transcripts), batch_size):
        batch = list(transcripts[start:start + batch_size])

        response = requests.post(
            url,
            json={
                "ids": batch
            },
            params={
                "external_db": "UniProtKB/Swiss-Prot"
            },
            headers={
                "Content-Type": "application/json"
            },
            timeout=60
        )

        response.raise_for_status()

        results = response.json()

        for transcript, refs in results.items():
            if refs:
                results_map[transcript] = refs[0]["primary_id"]
            else:
                results_map[transcript] = None

    return results_map





transcripts_hg38 = (
    mutpred_merge_removal_hg38["Feature"]
    .dropna()
    .unique()
)
uniprot_map_hg38 = get_uniprot_ids_batch(
    transcripts_hg38,
    genome="hg38"
)

mutpred_merge_removal_hg38["UniProt_ID"] = (
    mutpred_merge_removal_hg38["Feature"]
    .map(uniprot_map_hg38)
)

print(
    mutpred_merge_removal_hg38["UniProt_ID"].notna().sum(),
    "of",
    len(mutpred_merge_removal_hg38),
    "rows have a UniProt ID"
)







transcripts_hg37 = (
    mutpred_merge_removal_hg37["Feature"]
    .dropna()
    .unique()
)

uniprot_map_hg37 = {
    transcript: get_uniprot_id(
        transcript,
        genome="hg37"
    )
    for transcript in transcripts_hg37
}

mutpred_merge_removal_hg37["UniProt_ID"] = (
    mutpred_merge_removal_hg37["Feature"]
    .map(uniprot_map_hg37)
)



def remove_polyphen_training_variants(vep_df):
    # 1. Load your VEP dataset and the PolyPhen-2 training file

    train_df = pd.read_csv(os.path.join(mount_data,'training-2.2.2.tar.gz'),
            sep=r"\s+",
            header= None,
            names=["uniprot_id", "position", "ref_aa", "alt_aa"],
            compression='gzip',
            low_memory=False
        )
    # 2. Ensure gene symbols match (assuming you've mapped training uniprot_ids to gene SYMBOLS)
    # train_df['SYMBOL'] = train_df['uniprot_id'].map(your_mapping_dict)

    # 3. Perform the anti-join to drop training variants from your dataset
    merged = vep_df.merge(
        train_df,
        left_on=["SYMBOL", "Protein_position", "Ref_AA", "Alt_AA"],
        right_on=["SYMBOL", "position", "ref_aa", "alt_aa"],
        how="left",
        indicator=True,
    )

    filtered_df = merged[merged["_merge"] == "left_only"].drop(
        columns=["_merge", "position", "ref_aa", "alt_aa"]
    )
    
    return filtered_df
    

removed_polyphen_hg38 = remove_polyphen_training_variants(filtered_hg38)
removed_polyphen_hg37 = remove_polyphen_training_variants(XX)


#summarize and plot outputs:

def build_filter_summary_table(step_dataframes):
    rows = []

    for step_label, step_df in step_dataframes.items():
        
            rows.append({
                "FilterStep": step_label,
                "Assembly": "All",
                "UniqueGeneSymbol": step_df["Gene"].nunique(),
                "UniqueVariationID": step_df["Uploaded_variation"].nunique(),
            })

    return pd.DataFrame(rows)




def plot_filter_summary(df_summary, metric="UniqueVariationID"):
    """
    Plot a grouped bar chart comparing GRCh37 and GRCh38 counts per filter step
    with exact numerical values labeled above each bar.
    """
    # Reshape data for plotting side-by-side assembly bars per step
    pivot_df = df_summary.pivot(index="FilterStep", columns="Assembly", values=metric)

    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_order = pivot_df.index.tolist()
    pivot_df.reindex(plot_order).plot(kind="bar", ax=ax, width=0.75)
    
    ax.set_title(f"Filtering Summary by Assembly ({metric})", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Filter Step", fontsize=12)
    plt.xticks(rotation=35, ha="right")
    
    # Annotate exact numbers above each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="{:,.0f}", padding=3, fontsize=9)
        
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()








def main():
  
  
    vep_out_hg38_df = read_vep_output(os.path.join(mount_results,
    "clinvar_filtered_for_ensemblVEP_hg38_vep_output.txt"))

    vep_out_hg37_df = read_vep_output(os.path.join(mount_results,
    "clinvar_filtered_for_ensemblVEP_hg37_vep_output.txt"))
    
    mane_select_transcripts_hg38 = filter_mane_select_transcripts(vep_out_hg38_df)
    canonical_transcripts_hg37 = filter_canonical_transcripts(vep_out_hg37_df)

    
    vep_missense_hg38 = filter_missense_variants(mane_select_transcripts_hg38)
    vep_missense_hg37 = filter_missense_variants(canonical_transcripts_hg37)

    AF_filtered_hg38 = filter_AF_gnomad(vep_missense_hg38)  
    AF_filtered_hg37 = filter_AF_gnomad(vep_missense_hg37)  
    
    # Return a dictionary of the DataFrames you want to inspect
    return {
        "vep_out_hg38_df": vep_out_hg38_df,
        "vep_out_hg37_df": vep_out_hg37_df,
        "MANE_Select_hg38": mane_select_transcripts_hg38,
        "Canonical_hg37": canonical_transcripts_hg37,
        "AF_hg38": AF_filtered_hg38,
        "AF_hg37": AF_filtered_hg37,
        "missense_hg38": vep_missense_hg38,
        "missense_hg37": vep_missense_hg37
    }




# Execute main and capture outputs into global variables
if __name__ == "__main__":
    results = main()
    
    # Access and view any specific DataFrame
    vep_out_38 = results["vep_out_hg38_df"]
    vep_out_37 = results["vep_out_hg37_df"]
    af_38_df = results["AF_hg38"]
    af_37_df = results["AF_hg37"]
    missense_38_df = results["missense_hg38"]
    missense_37_df = results["missense_hg37"]
    mane_select_38_df = results["MANE_Select_hg38"]
    canonical_37_df = results["Canonical_hg37"] 
    
    
    


    
    
    
    
    