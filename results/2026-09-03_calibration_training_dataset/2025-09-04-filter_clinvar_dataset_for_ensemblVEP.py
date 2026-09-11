import sys
import pandas as pd 
import os
import re
import numpy as np
import matplotlib.pyplot as plt

#directories 
mount_data = "/Users/jwillis/minerva/pejaverlab/data/2026-09-03_calibration_training_dataset"
mount_results = "/Users/jwillis/minerva/pejaverlab/results/2026-09-03_calibration_training_dataset"

data= "/sc/arion/projects/pejaverlab/users/willij115/data/2026-09-03_calibration_training_dataset" 
results= "/sc/arion/projects/pejaverlab/users/willij115/results/2026-09-03_calibration_training_dataset"

#set pandas display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



# Load ClinVar variant summary data from the compressed text file
def load_clinvar_variant_summary(file_path):
    """
    Load ClinVar variant summary data from a compressed text file.

    Parameters:
    - file_path (str): Path to the compressed ClinVar variant summary file.

    Returns:
    - pd.DataFrame: DataFrame containing the ClinVar variant summary data.
    """
    return pd.read_csv(
        file_path,
        header=0,
        sep='\t',
        compression='gzip',
        low_memory=False
    )



##filtering 

#1. filter to only GRCh37 and GRCh38 assemblies
def filter_by_assembly(variant_df):
    """
    Filter the variant DataFrame to include only GRCh37 and GRCh38 assemblies.

    Parameters:
    - variant_df (pd.DataFrame): DataFrame containing variant data.

    Returns:
    - pd.DataFrame: Filtered DataFrame with only GRCh37 and GRCh38 assemblies.
    """
    assembly_mask = variant_df["Assembly"].isin(["GRCh37", "GRCh38"])
    return variant_df[assembly_mask]





#2. restrict to only pathogenic/likely pathogenic and benign/likely benign variants to remove ALL VUSs, and those with conflicting classifications

def filter_by_clinical_significance(hg38_hg37_filtered_df):
    """
    Filter the variant DataFrame to include only specific clinical significance categories.

    Parameters:
    - variant_df (pd.DataFrame): DataFrame containing variant data.

    Returns:
    - pd.DataFrame: Filtered DataFrame with only specified clinical significance categories.
    """
    print(hg38_hg37_filtered_df["ClinicalSignificance"].value_counts())
    
    clin_sign_words_to_keep = [
        "Pathogenic",
        "Likely pathogenic",
        "Pathogenic/Likely pathogenic",
        "Benign",
        "Likely benign",
        "Benign/Likely benign"
    ]
    clin_sig_mask = hg38_hg37_filtered_df["ClinicalSignificance"].isin(clin_sign_words_to_keep)
    return hg38_hg37_filtered_df.loc[clin_sig_mask].copy()




#3. keep only genes with atleast one pathogenic variant of any type 

def get_pathogenic_genes(clinical_significance_filtered_df):
    """
    Get unique gene symbols associated with pathogenic variants.

    Parameters:
    - clinical_significance_filtered_df (pd.DataFrame): DataFrame containing variant data.

    Returns:
    - np.ndarray: Array of unique gene symbols with pathogenic variants.
    """
    
    pathogenic_genes = (
    clinical_significance_filtered_df.loc[
        clinical_significance_filtered_df["ClinicalSignificance"].str.contains("Pathogenic", case=False, na=False),
        "GeneSymbol"
    ]
    .dropna()
    .unique())

    df_gene_filtered = clinical_significance_filtered_df[clinical_significance_filtered_df["GeneSymbol"].isin(pathogenic_genes)].copy()
   
    return df_gene_filtered


    




#4. remove all variants w/ zero-star review status(less than1 star),

def filter_by_review_status(atleast_one_pathogenic_gene_filtered_df):

    print(atleast_one_pathogenic_gene_filtered_df["ReviewStatus"].value_counts())

    list_review_status_to_remove = ["no assertion criteria provided", "no interpretation for the single variant" ]

    mask3 = atleast_one_pathogenic_gene_filtered_df["ReviewStatus"].isin(list_review_status_to_remove)

    return atleast_one_pathogenic_gene_filtered_df.loc[~mask3].copy()




# 5. Restrict to true single-nucleotide variants (SNVs) 
#    Both REF and ALT must be exactly one canonical nucleotide.

def filter_to_snvs(review_status_filtered_df):
    """
    Filter the variant DataFrame to include only true single-nucleotide variants (SNVs).

    Parameters:
    - review_status_filtered_df (pd.DataFrame): DataFrame containing variant data.

    Returns:
    - pd.DataFrame: Filtered DataFrame with only SNVs.
    """

    snv_mask = (
        review_status_filtered_df["ReferenceAllele"].str.fullmatch(
            r"[ACGT]", case=False, na=False
        )
        &
        review_status_filtered_df["AlternateAllele"].str.fullmatch(
            r"[ACGT]", case=False, na=False
        )
        &
        (
            review_status_filtered_df["ReferenceAllele"].str.upper()
            !=
            review_status_filtered_df["AlternateAllele"].str.upper()
        )
    )

    snv_filtered_df = review_status_filtered_df.loc[snv_mask].copy()

    print(
        f"Variants before SNV filtering: {len(review_status_filtered_df):,}"
    )
    print(
        f"SNVs after filtering: {len(snv_filtered_df):,}"
    )

    return snv_filtered_df




# 6. hg38 and hg37 filtered datasets for downstream processing with EnsemblVEP

def split_by_assembly(snv_filtered_df):
    """
    Split the SNV filtered DataFrame into separate DataFrames for GRCh38 and GRCh37 assemblies.

    Parameters:
    - snv_filtered_df (pd.DataFrame): DataFrame containing SNV filtered variant data.

    Returns:
    - tuple: Two DataFrames, one for GRCh38 and one for GRCh37.
    """
    hg38_filtered_df = snv_filtered_df[snv_filtered_df["Assembly"] == "GRCh38"].copy()
    hg37_filtered_df = snv_filtered_df[snv_filtered_df["Assembly"] == "GRCh37"].copy()
    
    return hg38_filtered_df, hg37_filtered_df




# 7. Create Ensembl VEP input for GRCh38

def create_vep_input(filtered_df):
    """
    Create a DataFrame suitable for Ensembl VEP input from the filtered variant DataFrame.

    Parameters:
    - filtered_df (pd.DataFrame): DataFrame containing filtered variant data.

    Returns:
    - pd.DataFrame: DataFrame formatted for Ensembl VEP input.
    """
    vep_input_df = pd.DataFrame({
        "chrom": filtered_df["Chromosome"],
        "start": filtered_df["Start"],
        "end": filtered_df["Start"],
        "allele": (
            filtered_df["ReferenceAllele"].str.upper()
            + "/"
            + filtered_df["AlternateAllele"].str.upper()
        ),
        "strand": "+",
        "id": filtered_df["VariationID"]
    })
    
    return vep_input_df





# 8. Write headerless files for Ensembl VEP

def write_vep_input_to_file(vep_input_df, output_path):
    """
    Write the Ensembl VEP input DataFrame to a headerless TSV file.

    Parameters:
    - vep_input_df (pd.DataFrame): DataFrame formatted for Ensembl VEP input.
    - output_path (str): Path to the output TSV file.
    """
    vep_input_df.to_csv(
        output_path,
        sep="\t",
        header=False,
        index=False
    )
    



# 9. Build a summary table of unique GeneSymbol and unique VariationID counts at each filter step


def build_filter_summary_table(step_dataframes, assembly_col="Assembly", assemblies=("GRCh37", "GRCh38")):
    """
    Build a summary table of unique GeneSymbol and unique VariationID counts
    at each step of the filtering pipeline, separated by genome assembly.
    """
    rows = []
    for step_label, step_df in step_dataframes.items():
        if assembly_col in step_df.columns:
            for asm in assemblies:
                asm_df = step_df[step_df[assembly_col] == asm]
                rows.append({
                    "FilterStep": step_label,
                    "Assembly": asm,
                    "UniqueGeneSymbol": asm_df["GeneSymbol"].nunique(),
                    "UniqueVariationID": asm_df["VariationID"].nunique(),
                })
        else:
            # Fallback if a step DataFrame lacks an Assembly column
            rows.append({
                "FilterStep": step_label,
                "Assembly": "All",
                "UniqueGeneSymbol": step_df["GeneSymbol"].nunique(),
                "UniqueVariationID": step_df["VariationID"].nunique(),
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
    pivot_df.plot(kind="bar", ax=ax, width=0.75)
    
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




# Pipeline Execution

def main():
    
    clinvar_raw_df = load_clinvar_variant_summary(os.path.join(mount_data,'variant_summary_2019-12.txt.gz'))
    
    assembly_filtered_df = filter_by_assembly(clinvar_raw_df)

    clin_sign_filtered_df = filter_by_clinical_significance(assembly_filtered_df)

    pathogenic_gene_filtered_df = get_pathogenic_genes(clin_sign_filtered_df)

    zero_review_filtered_df = filter_by_review_status(pathogenic_gene_filtered_df)

    snv_filtered_df = filter_to_snvs(zero_review_filtered_df)

    hg38_filtered_df, hg37_filtered_df = split_by_assembly(snv_filtered_df)

    vep_input_hg38 = create_vep_input(hg38_filtered_df)
    vep_input_hg37 = create_vep_input(hg37_filtered_df) 
            

    write_vep_input_to_file(
    vep_input_hg38,
    os.path.join(mount_results, "clinvar_filtered_for_ensemblVEP_hg38.tsv") 
    )

    write_vep_input_to_file(
        vep_input_hg37,
        os.path.join(mount_results, "clinvar_filtered_for_ensemblVEP_hg37.tsv")
    )       

    
    filter_summary_table = build_filter_summary_table({
        "1. Raw ClinVar variant summary": clinvar_raw_df,
        "2. Assembly filter (GRCh37/38)": assembly_filtered_df,
        "3. Clinical significance filter": clin_sign_filtered_df,
        "4. Gene has pathogenic variant filter": pathogenic_gene_filtered_df,
        "5. Review status filter (>=1 star)": zero_review_filtered_df,
        "6. SNV filter": snv_filtered_df,
        "7. GRCh38 subset": hg38_filtered_df,
        "8. GRCh37 subset": hg37_filtered_df,
    })

    # Display table
    print(filter_summary_table)

    # Plot VariationID counts (or change metric to "UniqueGeneSymbol")
    plot_filter_summary(filter_summary_table, metric="UniqueVariationID")
    plot_filter_summary(filter_summary_table, metric="UniqueGeneSymbol")
    
    filter_summary_table.to_csv(
        os.path.join(mount_results, "clinvar_filter_summary_table.tsv"),
        sep="\t",
        index=False
    )

if __name__ == "__main__":
    main()

