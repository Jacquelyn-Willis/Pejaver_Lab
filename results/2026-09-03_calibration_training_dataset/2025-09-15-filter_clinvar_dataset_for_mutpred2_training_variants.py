import sys
import pandas as pd 
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import requests
import tarfile
import time



#directories 
mount_data = "/Users/jwillis/minerva/pejaverlab/data/2026-09-03_calibration_training_dataset"
mount_results = "/Users/jwillis/minerva/pejaverlab/results/2026-09-03_calibration_training_dataset"

#data= 
#mount_data= "/sc/arion/projects/pejaverlab/users/willij115/data/2026-09-03_calibration_training_dataset" 

#results
#mount_results = "/sc/arion/projects/pejaverlab/users/willij115/results/2026-09-03_calibration_training_dataset"


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



##1.upload annotated clinvar datset 
needed_cols = [
    "Uploaded_variation",
    "Location",
    "Allele",
    "Gene",
    "Feature",
    "Feature_type",
    "Consequence",
    "cDNA_position",
    "CDS_position",
    "Protein_position",
    "Amino_acids",
    "Codons",
    "CANONICAL",
    "MANE",
    "MANE_SELECT",
    "TSL"
]

annot_clinvar_hg38 = pd.read_csv(
    os.path.join(
        mount_results,
        "annotated_variants_hg38_for_vep_training_filter.tsv"
    ),
    sep="\t",
    usecols=needed_cols
)
annot_clinvar_hg_38 = pd.read_csv(
        os.path.join(mount_results, 'annotated_variants_hg38_for_vep_training_filter.tsv'),
        sep="\t",
        header=0
        
    )


annot_clinvar_hg_37 = pd.read_csv(
        os.path.join(mount_results, 'annotated_variants_hg37_for_vep_training_filter.tsv'),
        sep="\t",
        header=0, 
        
    )



##2.upload mutpred2 variants and pivot long the variant column of the mutpred2 training variants 
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




##3. filter clinvar variant by the mutpred2 variants: 

#- normalize columns 



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



final1 = filter_clinvar_by_sequence_variant(
    annot_clinvar_hg_37,
    mutpred2_variants_df)









