import sys
import pandas as pd 
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import requests
import tarfile
import time
import openpyxl

#!{sys.executable} -m pip install openpyxl

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

uniprot_hg38 = pd.read_csv(os.path.join(mount_results, "clinvar_hg38_post_ensemble_vep_filters_w_uniprot.tsv"),
        sep="\t", header = 0, low_memory=False)

uniprot_hg37 = pd.read_csv(os.path.join(mount_results, "clinvar_hg37_post_ensemble_vep_filters_w_uniprot.tsv"),
        sep="\t", header = 0, low_memory=False)


mutpred2_entrezid_filtered_hg38 = pd.read_csv(os.path.join(mount_results, "clinvar_mutpred2_entrez_filtered_hg38.tsv"),
        sep="\t", header = 0,low_memory=False)

mutpred2_entrezid_filtered_hg37 = pd.read_csv(os.path.join(mount_results, "clinvar_mutpred2_entrez_filtered_hg37.tsv"),
        sep="\t", header = 0, low_memory=False)


mutpred2_prot_seq_filtered_hg38 = pd.read_csv(os.path.join(mount_results, "clinvar_mutpred2_prot_seq_filtered_hg38.tsv"),
        sep="\t", header = 0, low_memory=False)


mutpred2_prot_seq_filtered_hg37 = pd.read_csv(os.path.join(mount_results, "clinvar_mutpred2_prot_seq_filtered_hg37.tsv"),
        sep="\t", header = 0, low_memory=False)


#METHOD1:
##1. use uniprot ID from ensembl VEP output
##2. map them to polyphen2 uniporot ID and AA change and pos

#METHOD2:
##1. pull down uniprot ID from ensembl 
##2. map them to polyphen2 uniporot ID and AA change and pos

##POST ANALYSIS
##1. compare concordanance of methods 
#### which method allowed more accurate mapping ? 



#METHOD1: remove polyphen training variants that overlap using uniprot ID from ensembl VEP output
##***can only use method1 on hg38 because hg37 vep did not output the updated uniport annotation for hg37


#1. upload polyphen training set data
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


#2. filter and remove polyphen2 training variants using uniprot id and AA change and post 

def filter_clinvar_by_uniprot_training(
    clinvar_df,
    training_df,
    clinvar_uniprot_col="SWISSPROT_y",
    gene_col="SYMBOL",
    uploaded_variation_col="Uploaded_variation"
):
    """
    Filter a ClinVar dataframe using a UniProt-based MutPred2
    training dataframe.

    Matching key:
        normalized UniProt ID + position + ref_aa + alt_aa

    Example ClinVar:
        SWISSPROT_y      = P05161.238
        Protein_position = 141
        Amino_acids      = G/S

    Example training:
        uniprot_id = P05161
        position   = 141
        ref_aa     = G
        alt_aa     = S

    These are considered a match.

    Returns
    -------
    clinvar_filtered : pd.DataFrame
        Filtered ClinVar dataframe.

    summary : pd.DataFrame
        Before/after/removed counts.

    removed_variants : pd.DataFrame
        ClinVar rows removed because they matched the training set.
    """

    clinvar = clinvar_df.copy()
    training = training_df.copy()

    # =========================================================
    # 1. Normalize UniProt IDs
    # =========================================================

    clinvar["_uniprot_id"] = (
        clinvar[clinvar_uniprot_col]
        .astype("string")
        .str.strip()
        .str.split(".")
        .str[0]
    )

    training["_uniprot_id"] = (
        training["uniprot_id"]
        .astype("string")
        .str.strip()
        .str.split(".")
        .str[0]
    )

    # =========================================================
    # 2. Normalize protein positions
    # =========================================================

    clinvar["_position"] = pd.to_numeric(
        clinvar["Protein_position"],
        errors="coerce"
    ).astype("Int64")

    training["_position"] = pd.to_numeric(
        training["position"],
        errors="coerce"
    ).astype("Int64")

    # =========================================================
    # 3. Parse reference / alternate amino acids from ClinVar
    # =========================================================

    aa = (
        clinvar["Amino_acids"]
        .astype("string")
        .str.split("/", n=1, expand=True)
    )

    clinvar["_ref_aa"] = aa[0].str.strip()
    clinvar["_alt_aa"] = aa[1].str.strip()

    # =========================================================
    # 4. Normalize training amino acids
    # =========================================================

    training["_ref_aa"] = (
        training["ref_aa"]
        .astype("string")
        .str.strip()
    )

    training["_alt_aa"] = (
        training["alt_aa"]
        .astype("string")
        .str.strip()
    )

    # =========================================================
    # 5. Create explicit matching key
    #
    # Example:
    # P05161:G141S
    # =========================================================

    clinvar["uniprot_variant_key"] = (
        clinvar["_uniprot_id"]
        + ":"
        + clinvar["_ref_aa"]
        + clinvar["_position"].astype("string")
        + clinvar["_alt_aa"]
    )

    training["uniprot_variant_key"] = (
        training["_uniprot_id"]
        + ":"
        + training["_ref_aa"]
        + training["_position"].astype("string")
        + training["_alt_aa"]
    )

    # =========================================================
    # 6. Starting counts
    # =========================================================

    before_rows = len(clinvar)

    before_genes = clinvar[
        gene_col
    ].nunique(dropna=True)

    before_variants = clinvar[
        uploaded_variation_col
    ].nunique(dropna=True)

    # =========================================================
    # 7. Training-set keys
    # =========================================================

    training_keys = (
        training[
            ["uniprot_variant_key"]
        ]
        .dropna()
        .drop_duplicates()
        .assign(_training_match=True)
    )

    # =========================================================
    # 8. Match ClinVar to training set
    # =========================================================

    clinvar = clinvar.merge(
        training_keys,
        on="uniprot_variant_key",
        how="left"
    )

    # =========================================================
    # 9. Identify variants being removed
    # =========================================================

    removed_variants = clinvar[
        clinvar["_training_match"].eq(True)
    ].copy()

    # =========================================================
    # 10. Filter
    # =========================================================

    clinvar_filtered = clinvar[
        clinvar["_training_match"].isna()
    ].copy()

    # =========================================================
    # 11. Ending counts
    # =========================================================

    after_rows = len(clinvar_filtered)

    after_genes = clinvar_filtered[
        gene_col
    ].nunique(dropna=True)

    after_variants = clinvar_filtered[
        uploaded_variation_col
    ].nunique(dropna=True)

    # =========================================================
    # 12. Summary
    # =========================================================

    summary = pd.DataFrame({
        "metric": [
            "Rows",
            "Unique genes",
            "Unique Uploaded_variation"
        ],
        "before_filter": [
            before_rows,
            before_genes,
            before_variants
        ],
        "after_filter": [
            after_rows,
            after_genes,
            after_variants
        ],
        "removed": [
            before_rows - after_rows,
            before_genes - after_genes,
            before_variants - after_variants
        ]
    })

    # =========================================================
    # 13. Remove temporary columns
    # =========================================================

    helper_cols = [
        "_uniprot_id",
        "_position",
        "_ref_aa",
        "_alt_aa",
        "_training_match"
    ]

    clinvar_filtered = clinvar_filtered.drop(
        columns=helper_cols,
        errors="ignore"
    )

    removed_variants = removed_variants.drop(
        columns=helper_cols,
        errors="ignore"
    )

    return (
        clinvar_filtered,
        summary,
        removed_variants
    )



(
    clinvar_uniprot_filtered_w_mutpred2_entrezid_filtered_hg38,
    uniprot_summary_w_mutpred2_entrezid_filtered_hg38,
    uniprot_removed_w_mutpred2_entrezid_filtered_hg38
) = filter_clinvar_by_uniprot_training(
    clinvar_df=mutpred2_entrezid_filtered_hg38,
    training_df=poly_phen_training_set,
    clinvar_uniprot_col="SWISSPROT"
)

print(uniprot_summary_w_mutpred2_entrezid_filtered_hg38)

(
    clinvar_uniprot_filtered_w_mutpred2_prot_seq_filtered_hg38,
    uniprot_summary_w_mutpred2_prot_seq_filtered_hg38,
    uniprot_removed_w_mutpred2_prot_seq_filtered_hg38
) = filter_clinvar_by_uniprot_training(
    clinvar_df=mutpred2_prot_seq_filtered_hg38,
    training_df=poly_phen_training_set,
    clinvar_uniprot_col="SWISSPROT"
)

print(uniprot_summary_w_mutpred2_prot_seq_filtered_hg38)





def plot_filter_summary(summary, title="Training-set filtering summary"):

    plot_df = summary.set_index("metric")[
        ["before_filter", "after_filter", "removed"]
    ]

    ax = plot_df.plot(
        kind="bar",
        figsize=(10, 6),
        width=0.8
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_xticklabels(
        ["Rows", "Unique genes", "Unique Uploaded_variation"],
        rotation=0
    )

    # Add exact values above bars
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="{:,.0f}",
            padding=3
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

plot_filter_summary(uniprot_summary_w_mutpred2_entrezid_filtered_hg38)
plot_filter_summary(uniprot_summary_w_mutpred2_prot_seq_filtered_hg38)




##final output comparison


mcc2_df = pd.read_excel(os.path.join(mount_data, "mmc2-3.xlsx"))


import pandas as pd
import matplotlib.pyplot as plt


def compare_vep_mcc2_concordance(
    vep_df,
    mcc2_df,
    plot=True
):
    """
    Compare VEP and MCC2 datasets using:

        1. Gene
        2. rsID
        3. Gene + rsID

    VEP columns expected:
        SYMBOL
        Existing_variation

    MCC2 columns expected:
        genename
        rs_dbSNP151

    Returns
    -------
    summary : pd.DataFrame
        Counts of unique genes, unique rsIDs, and gene+rsID
        combinations in each dataset and their overlaps.

    overlap_data : dict
        Sets containing shared and dataset-specific genes,
        rsIDs, and gene+rsID combinations.
    """

    # =========================================================
    # 1. Make copies
    # =========================================================

    vep = vep_df.copy()
    mcc2 = mcc2_df.copy()

    # =========================================================
    # 2. Parse rsIDs
    # =========================================================

    vep["rsID"] = (
        vep["Existing_variation"]
        .astype("string")
        .str.extract(r"(rs\d+)", expand=False)
    )

    mcc2["rsID"] = (
        mcc2["rs_dbSNP151"]
        .astype("string")
        .str.extract(r"(rs\d+)", expand=False)
    )

    # =========================================================
    # 3. Clean gene names
    # =========================================================

    vep["SYMBOL"] = (
        vep["SYMBOL"]
        .astype("string")
        .str.strip()
    )

    mcc2["genename"] = (
        mcc2["genename"]
        .astype("string")
        .str.strip()
    )

    # =========================================================
    # 4. Create Gene + rsID key
    # =========================================================

    vep["concordance_key"] = (
        vep["SYMBOL"]
        + ":"
        + vep["rsID"]
    )

    mcc2["concordance_key"] = (
        mcc2["genename"]
        + ":"
        + mcc2["rsID"]
    )

    # =========================================================
    # 5. Unique genes
    # =========================================================

    vep_genes = set(
        vep["SYMBOL"]
        .dropna()
        .unique()
    )

    mcc2_genes = set(
        mcc2["genename"]
        .dropna()
        .unique()
    )

    shared_genes = vep_genes & mcc2_genes

    vep_only_genes = vep_genes - mcc2_genes

    mcc2_only_genes = mcc2_genes - vep_genes

    # =========================================================
    # 6. Unique rsIDs
    # =========================================================

    vep_rsids = set(
        vep["rsID"]
        .dropna()
        .unique()
    )

    mcc2_rsids = set(
        mcc2["rsID"]
        .dropna()
        .unique()
    )

    shared_rsids = vep_rsids & mcc2_rsids

    vep_only_rsids = vep_rsids - mcc2_rsids

    mcc2_only_rsids = mcc2_rsids - vep_rsids

    # =========================================================
    # 7. Unique Gene + rsID combinations
    # =========================================================

    vep_keys = set(
        vep["concordance_key"]
        .dropna()
        .unique()
    )

    mcc2_keys = set(
        mcc2["concordance_key"]
        .dropna()
        .unique()
    )

    shared_keys = vep_keys & mcc2_keys

    vep_only_keys = vep_keys - mcc2_keys

    mcc2_only_keys = mcc2_keys - vep_keys

    # =========================================================
    # 8. Summary table
    # =========================================================

    summary = pd.DataFrame({
        "Metric": [
            "Unique genes",
            "Unique rsIDs",
            "Unique gene + rsID"
        ],

        "VEP total": [
            len(vep_genes),
            len(vep_rsids),
            len(vep_keys)
        ],

        "MCC2 total": [
            len(mcc2_genes),
            len(mcc2_rsids),
            len(mcc2_keys)
        ],

        "Shared": [
            len(shared_genes),
            len(shared_rsids),
            len(shared_keys)
        ],

        "VEP only": [
            len(vep_only_genes),
            len(vep_only_rsids),
            len(vep_only_keys)
        ],

        "MCC2 only": [
            len(mcc2_only_genes),
            len(mcc2_only_rsids),
            len(mcc2_only_keys)
        ]
    })

    # =========================================================
    # 9. Plot
    # =========================================================

    if plot:

        # -----------------------------------------------------
        # Gene plot
        # -----------------------------------------------------

        gene_counts = [
            len(vep_genes),
            len(mcc2_genes),
            len(shared_genes),
            len(vep_only_genes),
            len(mcc2_only_genes)
        ]

        gene_labels = [
            "VEP total",
            "MCC2 total",
            "Shared",
            "VEP only",
            "MCC2 only"
        ]

        fig, ax = plt.subplots(figsize=(8, 6))

        bars = ax.bar(
            gene_labels,
            gene_counts
        )

        ax.bar_label(
            bars,
            labels=[f"{x:,}" for x in gene_counts],
            padding=3
        )

        ax.set_ylabel("Number of unique genes")
        ax.set_title("Gene Concordance: VEP vs MCC2")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.xticks(
            rotation=25,
            ha="right"
        )

        plt.tight_layout()
        plt.show()

        # -----------------------------------------------------
        # rsID plot
        # -----------------------------------------------------

        rsid_counts = [
            len(vep_rsids),
            len(mcc2_rsids),
            len(shared_rsids),
            len(vep_only_rsids),
            len(mcc2_only_rsids)
        ]

        rsid_labels = [
            "VEP total",
            "MCC2 total",
            "Shared",
            "VEP only",
            "MCC2 only"
        ]

        fig, ax = plt.subplots(figsize=(8, 6))

        bars = ax.bar(
            rsid_labels,
            rsid_counts
        )

        ax.bar_label(
            bars,
            labels=[f"{x:,}" for x in rsid_counts],
            padding=3
        )

        ax.set_ylabel("Number of unique rsIDs")
        ax.set_title("rsID Concordance: VEP vs MCC2")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.xticks(
            rotation=25,
            ha="right"
        )

        plt.tight_layout()
        plt.show()

        # -----------------------------------------------------
        # Gene + rsID plot
        # -----------------------------------------------------

        key_counts = [
            len(vep_keys),
            len(mcc2_keys),
            len(shared_keys),
            len(vep_only_keys),
            len(mcc2_only_keys)
        ]

        key_labels = [
            "VEP total",
            "MCC2 total",
            "Shared",
            "VEP only",
            "MCC2 only"
        ]

        fig, ax = plt.subplots(figsize=(8, 6))

        bars = ax.bar(
            key_labels,
            key_counts
        )

        ax.bar_label(
            bars,
            labels=[f"{x:,}" for x in key_counts],
            padding=3
        )

        ax.set_ylabel("Number of unique gene + rsID")
        ax.set_title("Gene + rsID Concordance: VEP vs MCC2")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.xticks(
            rotation=25,
            ha="right"
        )

        plt.tight_layout()
        plt.show()

    # =========================================================
    # 10. Return overlap sets
    # =========================================================

    overlap_data = {
        "shared_genes": shared_genes,
        "vep_only_genes": vep_only_genes,
        "mcc2_only_genes": mcc2_only_genes,

        "shared_rsids": shared_rsids,
        "vep_only_rsids": vep_only_rsids,
        "mcc2_only_rsids": mcc2_only_rsids,

        "shared_gene_rsid": shared_keys,
        "vep_only_gene_rsid": vep_only_keys,
        "mcc2_only_gene_rsid": mcc2_only_keys
    }

    return summary, overlap_data


summary1, overlap_data1 = compare_vep_mcc2_concordance(
    clinvar_uniprot_filtered_w_mutpred2_entrezid_filtered_hg38,
    mcc2_df
)



summary2, overlap_data2 = compare_vep_mcc2_concordance(
    clinvar_uniprot_filtered_w_mutpred2_prot_seq_filtered_hg38,
    mcc2_df
)











#MEthod 2: pullfrom 




def get_uniprot_ids_hg38(input_df, server="https://rest.ensembl.org"):
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    
    ensembl_genes = input_df["Gene"].dropna().unique()
    results = {}
    

    for i, gene_id in enumerate(ensembl_genes, 1):
        url = f"{server}/xrefs/id/{gene_id}"
        
        try:
            r = session.get(
                url,
                params={"external_db": "Uniprot/SWISSPROT"},
                timeout=30
            )

            if r.status_code == 200:
                xrefs = r.json()
                results[gene_id] = [
                    x["primary_id"] for x in xrefs
                    if "primary_id" in x
                ]
            else:
                results[gene_id] = []

        except requests.RequestException:
            results[gene_id] = []

        # small delay to avoid hammering the API
        time.sleep(0.05)

        if i % 100 == 0:
            print(f"{i}/{len(ensembl_genes)} completed")

    return results

def add_uniprot_ids_hg38(final_df):
    ensembl_genes = final_df["Gene"].dropna().unique()

    print(len(ensembl_genes))
    print(ensembl_genes[:10])

    uniprot_map_hg38 = get_uniprot_ids_hg38(final_df)

    final_df["UniProt_ID"] = final_df["Gene"].map(uniprot_map_hg38)

    return final_df


uniprot_map_entrezid_filtered_hg38 = add_uniprot_ids_hg38(mutpred2_entrezid_filtered_hg38)
uniprot_map_entrezid_filtered_hg38.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38.tsv"), sep="\t", index=False)
#clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38 = add_uniprot_ids_hg38(mutpred2_entrezid_filtered_hg38)


clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38 = add_uniprot_ids_hg38(mutpred2_prot_seq_filtered_hg38)
clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38.tsv"), sep="\t", index=False)


def get_uniprot_ids_hg37(input_df):
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    
    ensembl_genes = input_df["Gene"].dropna().unique()
    results = {}
    

    for i, gene_id in enumerate(ensembl_genes, 1):
        url = f"https://grch37.rest.ensembl.org/xrefs/id/{gene_id}"
        
        try:
            r = session.get(
                url,
                params={"external_db": "Uniprot/SWISSPROT"},
                timeout=30
            )

            if r.status_code == 200:
                xrefs = r.json()
                results[gene_id] = [
                    x["primary_id"] for x in xrefs
                    if "primary_id" in x
                ]
            else:
                results[gene_id] = []

        except requests.RequestException:
            results[gene_id] = []

        # small delay to avoid hammering the API
        time.sleep(0.05)

        if i % 100 == 0:
            print(f"{i}/{len(ensembl_genes)} completed")

    return results

def add_uniprot_ids_hg37(final_df):
    ensembl_genes = final_df["Gene"].dropna().unique()

    print(len(ensembl_genes))
    print(ensembl_genes[:10])

    uniprot_map_hg37 = get_uniprot_ids_hg38(final_df)

    final_df["UniProt_ID"] = final_df["Gene"].map(uniprot_map_hg37)

    return final_df


uniprot_map_entrezid_filtered_hg37 = add_uniprot_ids_hg37(mutpred2_entrezid_filtered_hg37)
uniprot_map_entrezid_filtered_hg37.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg37.tsv"), sep="\t", index=False)
#clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg37 = add_uniprot_ids_hg37(mutpred2_prot_seq_filtered_hg37)


clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37 = add_uniprot_ids_hg37(mutpred2_prot_seq_filtered_hg37)
clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37.tsv"), sep="\t", index=False)



'''

linvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38 = add_uniprot_ids_hg38(mutpred2_entrezid_filtered_hg38)
clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38.tsv"), sep="\t", index=False)

clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg37= add_uniprot_ids_hg37(mutpred2_entrezid_filtered_hg37)
clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg37.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg37.tsv"), sep="\t", index=False)


clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38 = add_uniprot_ids_hg38(mutpred2_prot_seq_filtered_hg38)
clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38.tsv"), sep="\t", index=False)

clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37= add_uniprot_ids_hg37(mutpred2_prot_seq_filtered_hg37)
clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37.tsv"), sep="\t", index=False)

'''
#METHOD2:


#1. get uniprot ID from ensemble rest


def get_uniprot_ids_hg38(input_df, server="https://rest.ensembl.org"):
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    
    ensembl_genes = input_df["Gene"].dropna().unique()
    results = {}
    

    for i, gene_id in enumerate(ensembl_genes, 1):
        url = f"{server}/xrefs/id/{gene_id}"
        
        try:
            r = session.get(
                url,
                params={"external_db": "Uniprot/SWISSPROT"},
                timeout=30
            )

            if r.status_code == 200:
                xrefs = r.json()
                results[gene_id] = [
                    x["primary_id"] for x in xrefs
                    if "primary_id" in x
                ]
            else:
                results[gene_id] = []

        except requests.RequestException:
            results[gene_id] = []

        # small delay to avoid hammering the API
        time.sleep(0.05)

        if i % 100 == 0:
            print(f"{i}/{len(ensembl_genes)} completed")

    return results

def add_uniprot_ids_hg38(final_df):
    ensembl_genes = final_df["Gene"].dropna().unique()

    print(len(ensembl_genes))
    print(ensembl_genes[:10])

    uniprot_map_hg38 = get_uniprot_ids_hg38(final_df)

    final_df["UniProt_ID"] = final_df["Gene"].map(uniprot_map_hg38)

    return final_df


uniprot_map_entrezid_filtered_hg38 = add_uniprot_ids_hg38(mutpred2_entrezid_filtered_hg38)
uniprot_map_entrezid_filtered_hg38.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38.tsv"), sep="\t", index=False)
#clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38 = add_uniprot_ids_hg38(mutpred2_prot_seq_filtered_hg38)


clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38 = add_uniprot_ids_hg38(mutpred2_prot_seq_filtered_hg38)
clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38.tsv"), sep="\t", index=False)


def get_uniprot_ids_hg37(input_df):
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    
    ensembl_genes = input_df["Gene"].dropna().unique()
    results = {}
    

    for i, gene_id in enumerate(ensembl_genes, 1):
        url = f"https://grch37.rest.ensembl.org/xrefs/id/{gene_id}"
        
        try:
            r = session.get(
                url,
                params={"external_db": "Uniprot/SWISSPROT"},
                timeout=30
            )

            if r.status_code == 200:
                xrefs = r.json()
                results[gene_id] = [
                    x["primary_id"] for x in xrefs
                    if "primary_id" in x
                ]
            else:
                results[gene_id] = []

        except requests.RequestException:
            results[gene_id] = []

        # small delay to avoid hammering the API
        time.sleep(0.05)

        if i % 100 == 0:
            print(f"{i}/{len(ensembl_genes)} completed")

    return results

def add_uniprot_ids_hg37(final_df):
    ensembl_genes = final_df["Gene"].dropna().unique()

    print(len(ensembl_genes))
    print(ensembl_genes[:10])

    uniprot_map_hg37 = get_uniprot_ids_hg38(final_df)

    final_df["UniProt_ID"] = final_df["Gene"].map(uniprot_map_hg37)

    return final_df


uniprot_map_entrezid_filtered_hg37 = add_uniprot_ids_hg37(mutpred2_entrezid_filtered_hg37)
uniprot_map_entrezid_filtered_hg37.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg37.tsv"), sep="\t", index=False)
#clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg37 = add_uniprot_ids_hg37(mutpred2_prot_seq_filtered_hg37)


clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37 = add_uniprot_ids_hg37(mutpred2_prot_seq_filtered_hg37)
clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37.tsv"), sep="\t", index=False)


#########################################

clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38 = add_uniprot_ids_hg38(mutpred2_entrezid_filtered_hg38)
clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38.tsv"), sep="\t", index=False)

clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg37= add_uniprot_ids_hg37(mutpred2_entrezid_filtered_hg37)
clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg37.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg37.tsv"), sep="\t", index=False)


clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38 = add_uniprot_ids_hg38(mutpred2_prot_seq_filtered_hg38)
clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38.tsv"), sep="\t", index=False)

clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37= add_uniprot_ids_hg37(mutpred2_prot_seq_filtered_hg37)
clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37.to_csv(os.path.join(mount_results, "clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37.tsv"), sep="\t", index=False)




#2. merge and remove overlap

clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38 = uniprot_map_entrezid_filtered_hg38.copy()

def filter_clinvar_by_uniprot_training(
    clinvar_df,
    training_df,
    clinvar_uniprot_col="SWISSPROT_y",
    gene_col="SYMBOL",
    uploaded_variation_col="Uploaded_variation"
):
    """
    Filter a ClinVar dataframe using a UniProt-based polyphen2
    training dataframe.

    Matching key:
        normalized UniProt ID + position + ref_aa + alt_aa

    Example ClinVar:
        SWISSPROT_y      = P05161.238
        Protein_position = 141
        Amino_acids      = G/S

    Example training:
        uniprot_id = P05161
        position   = 141
        ref_aa     = G
        alt_aa     = S

    These are considered a match.

    Returns
    -------
    clinvar_filtered : pd.DataFrame
        Filtered ClinVar dataframe.

    summary : pd.DataFrame
        Before/after/removed counts.

    removed_variants : pd.DataFrame
        ClinVar rows removed because they matched the training set.
    """

    clinvar = clinvar_df.copy()
    training = training_df.copy()

    # =========================================================
    # 1. Normalize UniProt IDs
    # =========================================================

    clinvar["_uniprot_id"] = (
        clinvar[clinvar_uniprot_col]
        .astype("string")
        .str.strip()
        .str.split(".")
        .str[0]
    )

    training["_uniprot_id"] = (
        training["uniprot_id"]
        .astype("string")
        .str.strip()
        .str.split(".")
        .str[0]
    )

    # =========================================================
    # 2. Normalize protein positions
    # =========================================================

    clinvar["_position"] = pd.to_numeric(
        clinvar["Protein_position"],
        errors="coerce"
    ).astype("Int64")

    training["_position"] = pd.to_numeric(
        training["position"],
        errors="coerce"
    ).astype("Int64")

    # =========================================================
    # 3. Parse reference / alternate amino acids from ClinVar
    # =========================================================

    aa = (
        clinvar["Amino_acids"]
        .astype("string")
        .str.split("/", n=1, expand=True)
    )

    clinvar["_ref_aa"] = aa[0].str.strip()
    clinvar["_alt_aa"] = aa[1].str.strip()

    # =========================================================
    # 4. Normalize training amino acids
    # =========================================================

    training["_ref_aa"] = (
        training["ref_aa"]
        .astype("string")
        .str.strip()
    )

    training["_alt_aa"] = (
        training["alt_aa"]
        .astype("string")
        .str.strip()
    )

    # =========================================================
    # 5. Create explicit matching key
    #
    # Example:
    # P05161:G141S
    # =========================================================

    clinvar["uniprot_variant_key"] = (
        clinvar["_uniprot_id"]
        + ":"
        + clinvar["_ref_aa"]
        + clinvar["_position"].astype("string")
        + clinvar["_alt_aa"]
    )

    training["uniprot_variant_key"] = (
        training["_uniprot_id"]
        + ":"
        + training["_ref_aa"]
        + training["_position"].astype("string")
        + training["_alt_aa"]
    )

    # =========================================================
    # 6. Starting counts
    # =========================================================

    before_rows = len(clinvar)

    before_genes = clinvar[
        gene_col
    ].nunique(dropna=True)

    before_variants = clinvar[
        uploaded_variation_col
    ].nunique(dropna=True)

    # =========================================================
    # 7. Training-set keys
    # =========================================================

    training_keys = (
        training[
            ["uniprot_variant_key"]
        ]
        .dropna()
        .drop_duplicates()
        .assign(_training_match=True)
    )

    # =========================================================
    # 8. Match ClinVar to training set
    # =========================================================

    clinvar = clinvar.merge(
        training_keys,
        on="uniprot_variant_key",
        how="left"
    )

    # =========================================================
    # 9. Identify variants being removed
    # =========================================================

    removed_variants = clinvar[
        clinvar["_training_match"].eq(True)
    ].copy()

    # =========================================================
    # 10. Filter
    # =========================================================

    clinvar_filtered = clinvar[
        clinvar["_training_match"].isna()
    ].copy()

    # =========================================================
    # 11. Ending counts
    # =========================================================

    after_rows = len(clinvar_filtered)

    after_genes = clinvar_filtered[
        gene_col
    ].nunique(dropna=True)

    after_variants = clinvar_filtered[
        uploaded_variation_col
    ].nunique(dropna=True)

    # =========================================================
    # 12. Summary
    # =========================================================

    summary = pd.DataFrame({
        "metric": [
            "Rows",
            "Unique genes",
            "Unique Uploaded_variation"
        ],
        "before_filter": [
            before_rows,
            before_genes,
            before_variants
        ],
        "after_filter": [
            after_rows,
            after_genes,
            after_variants
        ],
        "removed": [
            before_rows - after_rows,
            before_genes - after_genes,
            before_variants - after_variants
        ]
    })

    # =========================================================
    # 13. Remove temporary columns
    # =========================================================

    helper_cols = [
        "_uniprot_id",
        "_position",
        "_ref_aa",
        "_alt_aa",
        "_training_match"
    ]

    clinvar_filtered = clinvar_filtered.drop(
        columns=helper_cols,
        errors="ignore"
    )

    removed_variants = removed_variants.drop(
        columns=helper_cols,
        errors="ignore"
    )

    return (
        clinvar_filtered,
        summary,
        removed_variants
    )

(
    clinvar_rest_uniprot_filtered_w_mutpred2_entrezid_filtered_hg38,
    uniprot_rest_summary_w_mutpred2_entrezid_filtered_hg38,
    uniprot_rest_removed_w_mutpred2_entrezid_filtered_hg38
) = filter_clinvar_by_uniprot_training(
    clinvar_df=clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg38,
    training_df=poly_phen_training_set,
    clinvar_uniprot_col="SWISSPROT"
)

print(uniprot_rest_summary_w_mutpred2_entrezid_filtered_hg38)


(
    clinvar_rest_uniprot_filtered_w_mutpred2_entrezid_filtered_hg37,
    uniprot_rest_summary_w_mutpred2_entrezid_filtered_hg37,
    uniprot_rest_removed_w_mutpred2_entrezid_filtered_hg37
) = filter_clinvar_by_uniprot_training(
    clinvar_df=clinvar_w_rest_unipro_mutpred2_entrezid_filtered_hg37,
    training_df=poly_phen_training_set,
    clinvar_uniprot_col="Ensembl_UniProt_ID"
)

print(uniprot_rest_summary_w_mutpred2_entrezid_filtered_hg37)







(
    clinvar_rest_uniprot_filtered_w_mutpred2_prot_seq_filtered_hg38,
    uniprot_rest_summary_w_mutpred2_prot_seq_filtered_hg38,
    uniprot_rest_removed_w_mutpred2_prot_seq_filtered_hg38
) = filter_clinvar_by_uniprot_training(
    clinvar_df=clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg38,
    training_df=poly_phen_training_set,
    clinvar_uniprot_col="Ensembl_UniProt_ID"
)

print(uniprot_rest_summary_w_mutpred2_prot_seq_filtered_hg38)


(
    clinvar_rest_uniprot_filtered_w_mutpred2_prot_seq_filtered_hg37,
    uniprot_rest_summary_w_mutpred2_prot_seq_filtered_hg37,
    uniprot_rest_removed_w_mutpred2_prot_seq_filtered_hg37
) = filter_clinvar_by_uniprot_training(
    clinvar_df=clinvar_w_rest_unipro_mutpred2_prot_seq_filtered_hg37,
    training_df=poly_phen_training_set,
    clinvar_uniprot_col="Ensembl_UniProt_ID"
)

print(uniprot_rest_summary_w_mutpred2_prot_seq_filtered_hg37)













##concordanance and plotting
#can only compare using hg38 to hg38


def compare_uniprot_filter_concordance(
    original_df,
    vep_uniprot_filtered_df,
    ensembl_rest_uniprot_filtered_df,
    uploaded_variation_col="Uploaded_variation",
    gene_col="SYMBOL"
):
    """
    Compare concordance between two UniProt-based filters:

        1. UniProt IDs obtained from VEP
        2. UniProt IDs obtained from Ensembl REST

    The comparison is performed on Uploaded_variation.

    Parameters
    ----------
    original_df : pd.DataFrame
        The dataframe BEFORE either UniProt-based filter.

    vep_uniprot_filtered_df : pd.DataFrame
        Dataframe after filtering using UniProt IDs from VEP.

    ensembl_rest_uniprot_filtered_df : pd.DataFrame
        Dataframe after filtering using UniProt IDs from Ensembl REST.

    uploaded_variation_col : str
        Column identifying the variant.

    gene_col : str
        Gene column.

    Returns
    -------
    summary : pd.DataFrame
        Concordance summary.

    removed_by_vep : pd.DataFrame
        Variants removed only by the VEP-UniProt filter.

    removed_by_rest : pd.DataFrame
        Variants removed only by the Ensembl-REST-UniProt filter.

    removed_by_both : pd.DataFrame
        Variants removed by both filters.
    """

    # ---------------------------------------------------------
    # Create sets of Uploaded_variation
    # ---------------------------------------------------------

    original_variants = set(
        original_df[uploaded_variation_col]
        .dropna()
    )

    vep_remaining = set(
        vep_uniprot_filtered_df[uploaded_variation_col]
        .dropna()
    )

    rest_remaining = set(
        ensembl_rest_uniprot_filtered_df[uploaded_variation_col]
        .dropna()
    )

    # ---------------------------------------------------------
    # Variants removed by each filter
    # ---------------------------------------------------------

    vep_removed = (
        original_variants - vep_remaining
    )

    rest_removed = (
        original_variants - rest_remaining
    )

    # ---------------------------------------------------------
    # Concordance categories
    # ---------------------------------------------------------

    removed_by_both_set = (
        vep_removed & rest_removed
    )

    vep_only_set = (
        vep_removed - rest_removed
    )

    rest_only_set = (
        rest_removed - vep_removed
    )

    union_removed_set = (
        vep_removed | rest_removed
    )

    # ---------------------------------------------------------
    # Concordance calculations
    # ---------------------------------------------------------

    jaccard = (
        len(removed_by_both_set) /
        len(union_removed_set)
        if union_removed_set
        else 1.0
    )

    vep_shared = (
        len(removed_by_both_set) /
        len(vep_removed) * 100
        if vep_removed
        else 0
    )

    rest_shared = (
        len(removed_by_both_set) /
        len(rest_removed) * 100
        if rest_removed
        else 0
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = pd.DataFrame({
        "metric": [
            "Original Uploaded_variation",
            "Removed by VEP UniProt",
            "Removed by Ensembl REST UniProt",
            "Removed by both",
            "Removed only by VEP UniProt",
            "Removed only by Ensembl REST UniProt",
            "Union of variants removed",
            "Jaccard concordance (%)",
            "VEP UniProt removals shared with Ensembl REST (%)",
            "Ensembl REST UniProt removals shared with VEP (%)"
        ],
        "count": [
            len(original_variants),
            len(vep_removed),
            len(rest_removed),
            len(removed_by_both_set),
            len(vep_only_set),
            len(rest_only_set),
            len(union_removed_set),
            jaccard * 100,
            vep_shared,
            rest_shared
        ]
    })

    # ---------------------------------------------------------
    # Return actual variant rows for each category
    # ---------------------------------------------------------

    removed_by_vep = original_df[
        original_df[uploaded_variation_col].isin(
            vep_only_set
        )
    ].copy()

    removed_by_rest = original_df[
        original_df[uploaded_variation_col].isin(
            rest_only_set
        )
    ].copy()

    removed_by_both = original_df[
        original_df[uploaded_variation_col].isin(
            removed_by_both_set
        )
    ].copy()

    return (
        summary,
        removed_by_vep,
        removed_by_rest,
        removed_by_both
    )
    
    

(
    uniprot_concordance_summary_w_mutpred2_prot_seq_filtered_hg38,
    uniprot_vep_only_w_mutpred2_prot_seq_filtered_hg38,
    uniprot_rest_only_w_mutpred2_prot_seq_filtered_hg38,
    uniprot_both_w_mutpred2_prot_seq_filtered_hg38
) = compare_uniprot_filter_concordance(
    original_df=mutpred2_prot_seq_filtered_hg38,
    vep_uniprot_filtered_df=clinvar_uniprot_filtered_w_mutpred2_prot_seq_filtered_hg38,
    ensembl_rest_uniprot_filtered_df=clinvar_rest_uniprot_filtered_w_mutpred2_prot_seq_filtered_hg38
)

print(uniprot_concordance_summary_w_mutpred2_prot_seq_filtered_hg38)



(
    uniprot_concordance_summary_w_mutpred2_entrezid_filtered_hg38,
    uniprot_vep_only_w_mutpred2_entrezid_filtered_hg38,
    uniprot_rest_only_w_mutpred2_entrezid_filtered_hg38,
    uniprot_both_w_mutpred2_entrezid_filtered_hg38
) = compare_uniprot_filter_concordance(
    original_df=mutpred2_entrezid_filtered_hg38,
    vep_uniprot_filtered_df=clinvar_uniprot_filtered_w_mutpred2_entrezid_filtered_hg38,
    ensembl_rest_uniprot_filtered_df=clinvar_rest_uniprot_filtered_w_mutpred2_entrezid_filtered_hg38
)

print(uniprot_concordance_summary_w_mutpred2_entrezid_filtered_hg38)