import sys
import pandas as pd 
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import requests
import tarfile
import time
#!{sys.executable} -m pip install requests

#directories 
mount_data = "/Users/jwillis/minerva/pejaverlab/data/2026-09-03_calibration_training_dataset"
mount_results = "/Users/jwillis/minerva/pejaverlab/results/2026-09-03_calibration_training_dataset"

#data= 
#mount_data= "/sc/arion/projects/pejaverlab/users/willij115/data/2026-09-03_calibration_training_dataset" 

#results
#mount_results = "/sc/arion/projects/pejaverlab/users/willij115/results/2026-09-03_calibration_training_dataset"


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#LOAD DATA

input_hg38 = pd.read_csv(os.path.join(mount_results, "clinvar_hg38_post_ensemble_vep_filters_w_uniprot.tsv"),
        sep="\t", header = 0)

input_hg37 = pd.read_csv(os.path.join(mount_results, "clinvar_hg37_post_ensemble_vep_filters_w_uniprot.tsv"),
        sep="\t", header = 0)


#METHOD1:
##1. pull down entrez_ID from ensembl
##2. map them to mutpred entrez ID and AA change/variant
##3.filter out for variants found in mutpred2 training set

#METHID2:
##1. pull down ensembl protein seq
##2. map them to mutpred2 protein seq and AA change/variant
##3.filter out for variants found in mutpred2 training set


##POST ANALYSIS
##1. compare concordanance of methods 
#### which method allowed more accurate mapping ? 



#Method1

##1. pull down entrez ID for clinvar dataset


def get_entrez_id_hg38(gene_id):
    url = f"https://rest.ensembl.org/xrefs/id/{gene_id}"

    r = requests.get(
        url,
        params={"external_db": "EntrezGene"},
        headers={"Content-Type": "application/json"}
    )

    if r.status_code != 200:
        return None

    results = r.json()

    if len(results) == 0:
        return None

    return results[0]["primary_id"]


def add_entrez_ids_hg38(final_df):
    ensembl_gene = final_df["Gene"].dropna().unique()
    print(len(ensembl_gene))
    print(ensembl_gene[:10])

    entrez_map_hg38 = {}
    for gene1 in ensembl_gene:
        entrez_map_hg38[gene1] = get_entrez_id_hg38(gene1)
        time.sleep(0.1)  # stay under rate limit

    final_df["Entrez_ID"] = final_df["Gene"].map(entrez_map_hg38)
    return final_df



def get_entrez_id_hg37(gene_id):
    url = f"https://grch37.rest.ensembl.org/xrefs/id/{gene_id}"
    r = requests.get(
        url,
        params={"external_db": "EntrezGene"},
        headers={"Content-Type": "application/json"}
    )
    if r.status_code != 200:
        return None

    results = r.json()
    if not results:
        return None

    return results[0]["primary_id"]

def add_entrez_ids_hg37(final_df):
    ensembl_gene = final_df["Gene"].dropna().unique()
    print(len(ensembl_gene))
    print(ensembl_gene[:10])

    entrez_map_hg37 = {}
    for gene1 in ensembl_gene:
        entrez_map_hg37[gene1] = get_entrez_id_hg37(gene1)
        time.sleep(0.1)  # stay under rate limit

    final_df["Entrez_ID"] = final_df["Gene"].map(entrez_map_hg37)
    return final_df

#usager: 


hg_38_entrez_id_df = add_entrez_ids_hg38(input_hg38)
hg_38_entrez_id_df.to_csv(os.path.join(mount_results, "vep_output_w_entrez_id_hg38.tsv"), sep="\t", index=False)


hg_37_entrez_id_df = add_entrez_ids_hg37(input_hg37)
hg_37_entrez_id_df.to_csv(os.path.join(mount_results, "vep_output_w_entrez_id_hg37.tsv"), sep="\t", index=False)


##2.filter clinvar variants by mutpred2 training variants
def pivot_long_mutpred2_training_variants():
    train_df = pd.read_csv(
        os.path.join(mount_data, 'mp2_actual_training_data.txt'),
        sep="\t",
        header=None,
        low_memory=False
    )

    train_long = train_df[[1, 3, 4]].copy()

    train_long = train_long.rename(
        columns={
            1: "training_variants",
            3: "protein sequence",
            4: "Entrez_ID"
        }
    )

    # Split comma-separated variants into individual rows
    train_long["protein_variant"] = (
        train_long["training_variants"]
        .fillna("")
        .str.split(",")
    )

    train_long = train_long.explode("protein_variant")

    # Clean up
    train_long["protein_variant"] = (
        train_long["protein_variant"]
        .astype(str)
        .str.strip()
    )

    # Normalize Entrez IDs
    train_long["Entrez_ID"] = pd.to_numeric(
        train_long["Entrez_ID"],
        errors="coerce"
    ).astype("Int64")

    # Keep usable rows only
    train_long = train_long[
        train_long["Entrez_ID"].notna()
        & train_long["protein_variant"].notna()
        & (train_long["protein_variant"] != "")
        & (train_long["protein_variant"] != "nan")
    ].copy()

    # Only keep the columns needed for matching
    train_long = train_long[
        ["Entrez_ID", "protein_variant", "protein sequence"]
    ].drop_duplicates()

    return train_long


mutpred2_variants_df = pivot_long_mutpred2_training_variants()







#Method2

##1. pull down ensembl protein seq for clinvar dataset



def add_protein_sequences_hg38(final_df, id_column="Feature"):
    server = "https://rest.ensembl.org"
    ext = "/sequence/id"

    transcript_ids = final_df[id_column].dropna().unique().tolist()
    print(len(transcript_ids))
    print(transcript_ids[:10])

    seq_map = {}
    batch_size = 50

    for i in range(0, len(transcript_ids), batch_size):
        batch = transcript_ids[i:i + batch_size]

        r = requests.post(
            server + ext,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json={"ids": batch, "type": "protein"}
        )

        if r.status_code != 200:
            print(f"Batch starting at {i} failed with status {r.status_code}")
            continue

        results = r.json()
        for entry in results:
            seq_map[entry["query"]] = entry.get("seq")

        time.sleep(0.1)

    final_df["Protein_Sequence"] = final_df[id_column].map(seq_map)
    return final_df


def add_protein_sequences_hg37(final_df, id_column="Feature"):
    server = "https://grch37.rest.ensembl.org"
    ext = "/sequence/id"

    transcript_ids = final_df[id_column].dropna().unique().tolist()
    print(len(transcript_ids))
    print(transcript_ids[:10])

    seq_map = {}
    batch_size = 50

    for i in range(0, len(transcript_ids), batch_size):
        batch = transcript_ids[i:i + batch_size]

        r = requests.post(
            server + ext,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json={"ids": batch, "type": "protein"}
        )

        if r.status_code != 200:
            print(f"Batch starting at {i} failed with status {r.status_code}")
            continue

        results = r.json()
        for entry in results:
            seq_map[entry["query"]] = entry.get("seq")

        time.sleep(0.1)

    final_df["Protein_Sequence"] = final_df[id_column].map(seq_map)
    return final_df


hg_38_protein_seq_id_df = add_protein_sequences_hg38(input_hg38)
hg_38_protein_seq_id_df.to_csv(os.path.join(mount_results, "vep_output_w_protein_seq_hg38.tsv"), sep="\t", index=False)


hg_37_protein_seq_id_df = add_protein_sequences_hg37(input_hg37)
hg_37_protein_seq_id_df.to_csv(os.path.join(mount_results, "vep_output_w_protein_seq_hg37.tsv"), sep="\t", index=False)

















'''
# ============================================================
# 1. CREATE PROTEIN VARIANT ANNOTATION
# ============================================================

def convert_to_protein_variant_annot(df):
    df = df.copy()

    # Split S/N -> S and N
    df[["Ref_AA", "Alt_AA"]] = df["Amino_acids"].str.split(
        "/",
        expand=True
    )

    # Make sure position is numeric
    df["Protein_position"] = pd.to_numeric(
        df["Protein_position"],
        errors="coerce"
    )

    # Create S21N, G141S, R164P, etc.
    valid = (
        df["Ref_AA"].notna()
        & df["Alt_AA"].notna()
        & df["Protein_position"].notna()
    )

    df["protein_variant"] = pd.NA

    df.loc[valid, "protein_variant"] = (
        df.loc[valid, "Ref_AA"]
        + df.loc[valid, "Protein_position"]
            .astype(int)
            .astype(str)
        + df.loc[valid, "Alt_AA"]
    )

    return df


mutpred_to_merge_and_remove_hg38 = convert_to_protein_variant_annot(final_hg38)


mutpred_to_merge_and_remove_hg37 = (
    convert_to_protein_variant_annot(final_hg37)
)




# ============================================================
# 2. PREPARE MUTPRED2 TRAINING VARIANTS
# ============================================================

def prepare_mutpred2_training_variants():
    
    train_df = pd.read_csv(os.path.join(mount_data,'mp2_actual_training_data.txt'),
                    sep="\t",
                    header=None,
                    low_memory=False,
                    names=["dataset ID", "protein variants", "training labels", "protein sequence", "Entrez ID", "unknown" ],
                )

    train_long = train_df[[1, 4]].copy()

    train_long = train_long.rename(
        columns={
            1: "training_variants",
            4: "Entrez_ID"
        }
    )

    # Split comma-separated variants into individual rows
    train_long["protein_variant"] = (
        train_long["training_variants"]
        .fillna("")
        .str.split(",")
    )

    train_long = train_long.explode("protein_variant")

    # Clean up
    train_long["protein_variant"] = (
        train_long["protein_variant"]
        .astype(str)
        .str.strip()
    )

    # Normalize Entrez IDs
    train_long["Entrez_ID"] = pd.to_numeric(
        train_long["Entrez_ID"],
        errors="coerce"
    ).astype("Int64")

    # Keep usable rows only
    train_long = train_long[
        train_long["Entrez_ID"].notna()
        & train_long["protein_variant"].notna()
        & (train_long["protein_variant"] != "")
        & (train_long["protein_variant"] != "nan")
    ].copy()

    # Only keep the columns needed for matching
    train_long = train_long[
        ["Entrez_ID", "protein_variant"]
    ].drop_duplicates()
    
    return train_long


training_df_long = prepare_mutpred2_training_variants()

# ============================================================
# 3. REMOVE MUTPRED2 TRAINING OVERLAPS
# ============================================================

def remove_mutpred2_training_variants(df, train_long):

    df = df.copy()

    # Normalize Entrez ID
    df["Entrez_ID"] = pd.to_numeric(
        df["Entrez_ID"],
        errors="coerce"
    ).astype("Int64")

    merged = df.merge(
        train_long,
        on=["Entrez_ID", "protein_variant"],
        how="left",
        indicator=True
    )

    # Keep only variants NOT found in MutPred2 training data
    filtered_df = (
        merged[merged["_merge"] == "left_only"]
        .drop(columns=["_merge"])
        .copy()
    )

    return merged, filtered_df



# ============================================================
# 4. RUN HG38 / HG37
# ============================================================

mutpred_merge_removal_hg38, filtered_hg38 = (
    remove_mutpred2_training_variants(
        mutpred_to_merge_and_remove_hg38,
        training_df_long
    )
)

mutpred_merge_removal_hg37, filtered_hg37 = (
    remove_mutpred2_training_variants(
        mutpred_to_merge_and_remove_hg37,
        training_df_long
    )
)


#summary
def removal_summary(start_df, merged_df, filtered_df, label):
    
    # Rows that were removed because they matched MutPred2
    removed_df = merged_df[
        merged_df["_merge"] == "both"
    ].copy()

    print(f"\n{label}")
    print("=" * len(label))

    print(
        "Starting unique genes:",
        start_df["Gene"].nunique()
    )

    print(
        "Starting unique uploaded variations:",
        start_df["Uploaded_variation"].nunique()
    )

    print(
        "Removed unique genes:",
        removed_df["Gene"].nunique()
    )

    print(
        "Removed unique uploaded variations:",
        removed_df["Uploaded_variation"].nunique()
    )

    print(
        "Final unique genes:",
        filtered_df["Gene"].nunique()
    )

    print(
        "Final unique uploaded variations:",
        filtered_df["Uploaded_variation"].nunique()
    )


removal_summary(
    mutpred_to_merge_and_remove_hg38,
    mutpred_merge_removal_hg38,
    filtered_hg38,
    "HG38"
)

removal_summary(
    mutpred_to_merge_and_remove_hg37,
    mutpred_merge_removal_hg37,
    filtered_hg37,
    "HG37"
)

















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
    
    
    


    
    
    
    
'''