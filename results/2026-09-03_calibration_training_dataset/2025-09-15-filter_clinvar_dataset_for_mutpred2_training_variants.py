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


#hg_38_entrez_id_df = add_entrez_ids_hg38(input_hg38)
#hg_38_entrez_id_df.to_csv(os.path.join(mount_results, "vep_output_w_entrez_id_hg38.tsv"), sep="\t", index=False)


#hg_37_entrez_id_df = add_entrez_ids_hg37(input_hg37)
#hg_37_entrez_id_df.to_csv(os.path.join(mount_results, "vep_output_w_entrez_id_hg37.tsv"), sep="\t", index=False)




##2.reupload for usage

hg_38_entrez_id_df = pd.read_csv(
        os.path.join(mount_results, 'vep_output_w_entrez_id_hg38.tsv'),
        sep="\t",
        header=0,
        low_memory=False
    )

hg_37_entrez_id_df = pd.read_csv(
        os.path.join(mount_results, 'vep_output_w_entrez_id_hg37.tsv'),
        sep="\t",
        header=0,
        low_memory=False
    )



##3.pivot long the variant column of the mutpred2 training variants 
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


##4. filter clinvar variant by the mutpred2 variants

def filter_clinvar_by_entrez_variant(
    clinvar_df,
    mutpred2_variants_df,
    gene_col="SYMBOL",
    uploaded_variation_col="Uploaded_variation"
):
    """
    Remove ClinVar variants overlapping MutPred2 training data
    using Entrez_ID + protein_variant.

    A protein_variant key is constructed in ClinVar from:
        Amino_acids + Protein_position

    Example:
        Amino_acids = "A/V"
        Protein_position = 209
        -> protein_variant = "A209V"

    Returns
    -------
    clinvar_filtered : pd.DataFrame
        Filtered ClinVar dataframe.

    summary : pd.DataFrame
        Counts of unique genes and Uploaded_variation before and
        after filtering.
    """

    clinvar = clinvar_df.copy()
    mutpred2 = mutpred2_variants_df.copy()

    # ---------------------------------------------------------
    # Normalize Entrez IDs
    # ---------------------------------------------------------
    clinvar["Entrez_ID"] = pd.to_numeric(
        clinvar["Entrez_ID"],
        errors="coerce"
    ).astype("Int64")

    mutpred2["Entrez_ID"] = pd.to_numeric(
        mutpred2["Entrez_ID"],
        errors="coerce"
    ).astype("Int64")

    # ---------------------------------------------------------
    # Construct protein_variant in ClinVar
    # ---------------------------------------------------------
    aa = clinvar["Amino_acids"].fillna("").str.split(
        "/", expand=True
    )

    clinvar["protein_variant"] = (
        aa[0].str.strip()
        + clinvar["Protein_position"].astype("Int64").astype(str)
        + aa[1].str.strip()
    )

    # Invalid/missing variants -> NA
    invalid = (
        (aa[0].str.strip() == "") |
        (aa[1].str.strip() == "") |
        clinvar["Protein_position"].isna()
    )

    clinvar.loc[invalid, "protein_variant"] = pd.NA

    # ---------------------------------------------------------
    # Starting counts
    # ---------------------------------------------------------
    start_gene_count = clinvar[gene_col].nunique(dropna=True)
    start_variant_count = clinvar[uploaded_variation_col].nunique(
        dropna=True
    )
    start_row_count = len(clinvar)

    # ---------------------------------------------------------
    # Build MutPred2 matching keys
    # ---------------------------------------------------------
    mutpred2_keys = (
        mutpred2[
            ["Entrez_ID", "protein_variant"]
        ]
        .dropna()
        .drop_duplicates()
    )

    # ---------------------------------------------------------
    # Match
    # ---------------------------------------------------------
    clinvar = clinvar.merge(
        mutpred2_keys.assign(_mutpred2_match=True),
        on=["Entrez_ID", "protein_variant"],
        how="left"
    )

    # ---------------------------------------------------------
    # Remove overlapping variants
    # ---------------------------------------------------------
    clinvar_filtered = clinvar[
        clinvar["_mutpred2_match"].isna()
    ].copy()

    clinvar_filtered = clinvar_filtered.drop(
        columns="_mutpred2_match"
    )

    # ---------------------------------------------------------
    # Ending counts
    # ---------------------------------------------------------
    end_gene_count = clinvar_filtered[gene_col].nunique(
        dropna=True
    )
    end_variant_count = clinvar_filtered[
        uploaded_variation_col
    ].nunique(dropna=True)
    end_row_count = len(clinvar_filtered)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    summary = pd.DataFrame({
        "metric": [
            "Rows",
            "Unique genes",
            "Unique Uploaded_variation"
        ],
        "before_filter": [
            start_row_count,
            start_gene_count,
            start_variant_count
        ],
        "after_filter": [
            end_row_count,
            end_gene_count,
            end_variant_count
        ]
    })

    return clinvar_filtered, summary







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


#hg_38_protein_seq_id_df = add_protein_sequences_hg38(input_hg38)
#hg_38_protein_seq_id_df.to_csv(os.path.join(mount_results, "vep_output_w_protein_seq_hg38.tsv"), sep="\t", index=False)


#hg_37_protein_seq_id_df = add_protein_sequences_hg37(input_hg37)
#hg_37_protein_seq_id_df.to_csv(os.path.join(mount_results, "vep_output_w_protein_seq_hg37.tsv"), sep="\t", index=False)

##2.reupload dataframe for easy usage

hg_38_protein_seq_id_df = pd.read_csv(
        os.path.join(mount_results, 'vep_output_w_protein_seq_hg38.tsv'),
        sep="\t",
        header=0,
        low_memory=False
    )

hg_37_protein_seq_id_df = pd.read_csv(
        os.path.join(mount_results, 'vep_output_w_protein_seq_hg37.tsv'),
        sep="\t",
        header=0,
        low_memory=False
    )



#3. filter clinvar variants by protein seq and aa change 

def filter_clinvar_by_sequence_variant(
    clinvar_df,
    mutpred2_variants_df,
    clinvar_sequence_col="protein sequence",
    mutpred2_sequence_col="protein sequence",
    gene_col="SYMBOL",
    uploaded_variation_col="Uploaded_variation"
):
    """
    Remove ClinVar variants overlapping MutPred2 training data
    using protein sequence + protein_variant.

    Returns
    -------
    clinvar_filtered : pd.DataFrame
        Filtered ClinVar dataframe.

    summary : pd.DataFrame
        Counts of unique genes and Uploaded_variation before and
        after filtering.
    """

    clinvar = clinvar_df.copy()
    mutpred2 = mutpred2_variants_df.copy()

    # ---------------------------------------------------------
    # Construct protein_variant in ClinVar
    # ---------------------------------------------------------
    aa = clinvar["Amino_acids"].fillna("").str.split(
        "/", expand=True
    )

    clinvar["protein_variant"] = (
        aa[0].str.strip()
        + clinvar["Protein_position"].astype("Int64").astype(str)
        + aa[1].str.strip()
    )

    invalid = (
        (aa[0].str.strip() == "") |
        (aa[1].str.strip() == "") |
        clinvar["Protein_position"].isna()
    )

    clinvar.loc[invalid, "protein_variant"] = pd.NA

    # ---------------------------------------------------------
    # Normalize sequences
    # ---------------------------------------------------------
    clinvar[clinvar_sequence_col] = (
        clinvar[clinvar_sequence_col]
        .astype("string")
        .str.strip()
    )

    mutpred2[mutpred2_sequence_col] = (
        mutpred2[mutpred2_sequence_col]
        .astype("string")
        .str.strip()
    )

    # ---------------------------------------------------------
    # Starting counts
    # ---------------------------------------------------------
    start_gene_count = clinvar[gene_col].nunique(dropna=True)
    start_variant_count = clinvar[uploaded_variation_col].nunique(
        dropna=True
    )
    start_row_count = len(clinvar)

    # ---------------------------------------------------------
    # Build MutPred2 matching keys
    # ---------------------------------------------------------
    mutpred2_keys = (
        mutpred2[
            [mutpred2_sequence_col, "protein_variant"]
        ]
        .dropna()
        .drop_duplicates()
        .rename(
            columns={
                mutpred2_sequence_col: "_protein_sequence"
            }
        )
    )

    # Rename ClinVar sequence to common merge key
    clinvar["_protein_sequence"] = clinvar[
        clinvar_sequence_col
    ]

    # ---------------------------------------------------------
    # Match on sequence + protein_variant
    # ---------------------------------------------------------
    clinvar = clinvar.merge(
        mutpred2_keys.assign(_mutpred2_match=True),
        on=["_protein_sequence", "protein_variant"],
        how="left"
    )

    # ---------------------------------------------------------
    # Remove overlapping variants
    # ---------------------------------------------------------
    clinvar_filtered = clinvar[
        clinvar["_mutpred2_match"].isna()
    ].copy()

    # Remove helper columns
    clinvar_filtered = clinvar_filtered.drop(
        columns=[
            "_mutpred2_match",
            "_protein_sequence"
        ]
    )

    # ---------------------------------------------------------
    # Ending counts
    # ---------------------------------------------------------
    end_gene_count = clinvar_filtered[gene_col].nunique(
        dropna=True
    )
    end_variant_count = clinvar_filtered[
        uploaded_variation_col
    ].nunique(dropna=True)
    end_row_count = len(clinvar_filtered)

    summary = pd.DataFrame({
        "metric": [
            "Rows",
            "Unique genes",
            "Unique Uploaded_variation"
        ],
        "before_filter": [
            start_row_count,
            start_gene_count,
            start_variant_count
        ],
        "after_filter": [
            end_row_count,
            end_gene_count,
            end_variant_count
        ]
    })

    return clinvar_filtered, summary









##COMPARE THE TWO METHODS

def compare_filter_concordance(
    clinvar_original,
    clinvar_entrez_filtered,
    clinvar_sequence_filtered,
    uploaded_variation_col="Uploaded_variation",
    gene_col="SYMBOL"
):
    """
    Compare the variants removed by two filtering approaches.

    The two filtering approaches are assumed to have started from
    the same original ClinVar dataframe.

    Returns
    -------
    concordance_summary : pd.DataFrame
        Summary statistics comparing the two filters.

    removed_by_entrez : pd.DataFrame
        Variants removed only by Entrez_ID + protein_variant filter.

    removed_by_sequence : pd.DataFrame
        Variants removed only by sequence + protein_variant filter.

    common_removed : pd.DataFrame
        Variants removed by both filters.

    details : dict
        Sets containing the Uploaded_variation identifiers.
    """

    # ---------------------------------------------------------
    # Sets of variants in each dataset
    # ---------------------------------------------------------
    original = set(
        clinvar_original[uploaded_variation_col]
        .dropna()
    )

    entrez_remaining = set(
        clinvar_entrez_filtered[uploaded_variation_col]
        .dropna()
    )

    sequence_remaining = set(
        clinvar_sequence_filtered[uploaded_variation_col]
        .dropna()
    )

    # ---------------------------------------------------------
    # Variants removed by each filter
    # ---------------------------------------------------------
    removed_by_entrez_set = (
        original - entrez_remaining
    )

    removed_by_sequence_set = (
        original - sequence_remaining
    )

    common_removed_set = (
        removed_by_entrez_set &
        removed_by_sequence_set
    )

    entrez_only_set = (
        removed_by_entrez_set -
        removed_by_sequence_set
    )

    sequence_only_set = (
        removed_by_sequence_set -
        removed_by_entrez_set
    )

    # ---------------------------------------------------------
    # Concordance statistics
    # ---------------------------------------------------------
    union_removed = (
        removed_by_entrez_set |
        removed_by_sequence_set
    )

    jaccard = (
        len(common_removed_set) / len(union_removed)
        if union_removed
        else 1.0
    )

    entrez_shared_pct = (
        len(common_removed_set) /
        len(removed_by_entrez_set) * 100
        if removed_by_entrez_set
        else 0
    )

    sequence_shared_pct = (
        len(common_removed_set) /
        len(removed_by_sequence_set) * 100
        if removed_by_sequence_set
        else 0
    )

    # ---------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------
    concordance_summary = pd.DataFrame({
        "metric": [
            "Original Uploaded_variation",
            "Removed by Entrez + protein_variant",
            "Removed by sequence + protein_variant",
            "Removed by both",
            "Removed only by Entrez + protein_variant",
            "Removed only by sequence + protein_variant",
            "Union of variants removed",
            "Jaccard concordance (%)",
            "Entrez removals shared with sequence (%)",
            "Sequence removals shared with Entrez (%)"
        ],
        "count": [
            len(original),
            len(removed_by_entrez_set),
            len(removed_by_sequence_set),
            len(common_removed_set),
            len(entrez_only_set),
            len(sequence_only_set),
            len(union_removed),
            jaccard * 100,
            entrez_shared_pct,
            sequence_shared_pct
        ]
    })

    # ---------------------------------------------------------
    # Pull actual rows for each category
    # ---------------------------------------------------------
    removed_by_entrez = clinvar_original[
        clinvar_original[uploaded_variation_col].isin(
            entrez_only_set
        )
    ].copy()

    removed_by_sequence = clinvar_original[
        clinvar_original[uploaded_variation_col].isin(
            sequence_only_set
        )
    ].copy()

    common_removed = clinvar_original[
        clinvar_original[uploaded_variation_col].isin(
            common_removed_set
        )
    ].copy()

    details = {
        "removed_by_entrez": removed_by_entrez_set,
        "removed_by_sequence": removed_by_sequence_set,
        "common_removed": common_removed_set,
        "entrez_only": entrez_only_set,
        "sequence_only": sequence_only_set
    }

    return (
        concordance_summary,
        removed_by_entrez,
        removed_by_sequence,
        common_removed,
        details
    )






# Filter 1
clinvar_entrez_filtered_hg38, entrez_summary_hg38 = (
    filter_clinvar_by_entrez_variant(
        hg_38_entrez_id_df,
        mutpred2_variants_df
    )
)

clinvar_entrez_filtered_hg38.to_csv(os.path.join(mount_results, "clinvar_mutpred2_entrez_filtered_hg38.tsv"), sep="\t", index=False)
print(entrez_summary_hg38)    


clinvar_entrez_filtered_hg37, entrez_summary_hg37 = (
    filter_clinvar_by_entrez_variant(
        hg_37_entrez_id_df,
        mutpred2_variants_df
    )
)
clinvar_entrez_filtered_hg37.to_csv(os.path.join(mount_results, "clinvar_mutpred2_entrez_filtered_hg37.tsv"), sep="\t", index=False)

print(entrez_summary_hg37)    




# Filter 2
clinvar_sequence_filtered_hg38, sequence_summary_hg38 = (
    filter_clinvar_by_sequence_variant(
        hg_38_protein_seq_id_df,
        mutpred2_variants_df,
        clinvar_sequence_col="Protein_Sequence"
    )
)
clinvar_sequence_filtered_hg38.to_csv(os.path.join(mount_results, "clinvar_mutpred2_prot_seq_filtered_hg38.tsv"), sep="\t", index=False)


print(sequence_summary_hg38)


clinvar_sequence_filtered_hg37, sequence_summary_hg37 = (
    filter_clinvar_by_sequence_variant(
        hg_37_protein_seq_id_df,
        mutpred2_variants_df,
        clinvar_sequence_col="Protein_Sequence"
    )
)
clinvar_sequence_filtered_hg37.to_csv(os.path.join(mount_results, "clinvar_mutpred2_prot_seq_filtered_hg37.tsv"), sep="\t", index=False)

print(sequence_summary_hg37)


# Compare the two approaches
(
    concordance_summary_hg38,
    removed_entrez_only_hg38,
    removed_sequence_only_hg38,
    removed_common_hg38,
    concordance_details_hg38
) = compare_filter_concordance(
    clinvar_original=input_hg38,
    clinvar_entrez_filtered=clinvar_entrez_filtered_hg38,
    clinvar_sequence_filtered=clinvar_sequence_filtered_hg38
)

(
    concordance_summary_hg37,
    removed_entrez_only_hg37,
    removed_sequence_only_hg37,
    removed_common_hg37,
    concordance_details_hg37
) = compare_filter_concordance(
    clinvar_original=input_hg37,
    clinvar_entrez_filtered=clinvar_entrez_filtered_hg37,
    clinvar_sequence_filtered=clinvar_sequence_filtered_hg37
)

print("ENTREZ FILTER")
print(entrez_summary_hg38)

print("\nSEQUENCE FILTER")
print(sequence_summary_hg38)


print("ENTREZ FILTER")
print(entrez_summary_hg37)

print("\nSEQUENCE FILTER")
print(sequence_summary_hg37)

print("\nCONCORDANCE")
print(concordance_summary_hg38)

print("\nCONCORDANCE")
print(concordance_summary_hg37)



#plotting
def plot_filter_summaries(
    entrez_summary,
    sequence_summary,
    concordance_summary
):
    """
    Plot summary tables returned by the ClinVar/MutPred2 filtering functions.

    Parameters
    ----------
    entrez_summary : pd.DataFrame
        Summary returned by filter_clinvar_by_entrez_variant()

    sequence_summary : pd.DataFrame
        Summary returned by filter_clinvar_by_sequence_variant()

    concordance_summary : pd.DataFrame
        Summary returned by compare_filter_concordance()

    Returns
    -------
    None
        Displays three plots.
    """

    # ========================================================
    # Plot 1: Before vs After filtering
    # ========================================================

    # Combine the two filtering summaries
    metrics = entrez_summary["metric"].tolist()

    before = entrez_summary["before_filter"].values
    entrez_after = entrez_summary["after_filter"].values
    sequence_after = sequence_summary["after_filter"].values

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width,
        before,
        width,
        label="Before filter"
    )

    bars2 = ax.bar(
        x,
        entrez_after,
        width,
        label="Entrez ID + protein variant"
    )

    bars3 = ax.bar(
        x + width,
        sequence_after,
        width,
        label="Protein sequence + protein variant"
    )

    ax.set_ylabel("Count")
    ax.set_title(
        "ClinVar Variants Before and After MutPred2 Filtering"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    ax.legend(frameon=False)

    ax.bar_label(
        bars1,
        labels=[f"{int(v):,}" for v in before],
        padding=3
    )

    ax.bar_label(
        bars2,
        labels=[f"{int(v):,}" for v in entrez_after],
        padding=3
    )

    ax.bar_label(
        bars3,
        labels=[f"{int(v):,}" for v in sequence_after],
        padding=3
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


    # ========================================================
    # Plot 2: Concordance of variants removed
    # ========================================================

    # Pull values directly from concordance_summary
    concordance_lookup = dict(
        zip(
            concordance_summary["metric"],
            concordance_summary["count"]
        )
    )

    labels = [
        "Both filters",
        "Entrez only",
        "Sequence only"
    ]

    values = [
        concordance_lookup[
            "Removed by both"
        ],

        concordance_lookup[
            "Removed only by Entrez + protein_variant"
        ],

        concordance_lookup[
            "Removed only by sequence + protein_variant"
        ]
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(
        labels,
        values
    )

    ax.set_ylabel("Unique Uploaded_variation")
    ax.set_title(
        "Concordance of Variants Removed"
    )

    ax.bar_label(
        bars,
        labels=[f"{int(v):,}" for v in values],
        padding=4
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


    # ========================================================
    # Plot 3: Concordance percentages
    # ========================================================

    percentage_metrics = [
        "Jaccard concordance (%)",
        "Entrez removals shared with sequence (%)",
        "Sequence removals shared with Entrez (%)"
    ]

    percentage_labels = [
        "Jaccard",
        "Entrez → Sequence",
        "Sequence → Entrez"
    ]

    percentages = [
        concordance_lookup[m]
        for m in percentage_metrics
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(
        percentage_labels,
        percentages
    )

    ax.set_ylabel("Concordance (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Concordance Between Filtering Strategies"
    )

    ax.bar_label(
        bars,
        labels=[f"{v:.1f}%" for v in percentages],
        padding=4
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
    









#function calls 


plot_filter_summaries(
    entrez_summary=entrez_summary_hg38,
    sequence_summary=sequence_summary_hg38,
    concordance_summary=concordance_summary_hg38
)

plot_filter_summaries(
    entrez_summary=entrez_summary_hg37,
    sequence_summary=sequence_summary_hg37,
    concordance_summary=concordance_summary_hg37
)