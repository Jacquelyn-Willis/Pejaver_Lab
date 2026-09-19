import sys
import pandas as pd 
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import requests
import tarfile
import os
import pandas as pd
from Bio import SeqIO
import gzip

#!{sys.executable} -m pip install requests
#!{sys.executable} -m pip install Bio

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









#get clinvar variants protein seq and uniprot ID


##1. LOAD DATA
def fasta_to_df(fasta_file):
    records = []

    with gzip.open(fasta_file, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):

            fields = {
                x.split(":", 1)[0]: x.split(":", 1)[1]
                for x in record.description.split()
                if ":" in x
            }

            records.append({
                "gene_stable_id": fields["gene"].split(".")[0],
                "transcript_stable_id": fields["transcript"].split(".")[0],
                "protein_stable_id": record.id.split(".")[0],
                "sequence": str(record.seq)
            })

    return pd.DataFrame(records)




variants_hg38 = pd.read_csv(os.path.join(mount_results, "clinvar_hg38_post_ensemble_vep_filters_no_uniprot.tsv"), sep="\t", header = 0)
variants_hg37 = pd.read_csv(os.path.join(mount_results, "clinvar_hg37_post_ensemble_vep_filters_no_uniprot.tsv"), sep="\t", header = 0)


uniprot_df_hg38 = pd.read_csv(os.path.join(mount_data, "Homo_sapiens.GRCh38.116.uniprot.tsv.gz"),  sep="\t", low_memory=False )
prot_seq_df_hg38 = fasta_to_df(os.path.join(mount_data, "Homo_sapiens.GRCh38.pep.all.fa.gz"))


uniprot_df_hg37 = pd.read_csv(os.path.join(mount_data, "Homo_sapiens.GRCh37.85.uniprot.tsv.gz"),  sep="\t", low_memory=False)
prot_seq_df_hg37 = fasta_to_df(os.path.join(mount_data, "Homo_sapiens.GRCh37.pep.all.fa.gz"))




##2. MERGE uniprot and fsta protein sequences
def merge_uniprot_protein_seq(uniprot_df, prot_seq_df):
    
    merged_df = uniprot_df.merge(
        prot_seq_df,
        on=[
            "gene_stable_id",
            "transcript_stable_id",
            "protein_stable_id"
        ],
        how="right"
    )
    
    return merged_df


annot_df_hg37 = merge_uniprot_protein_seq(
    uniprot_df_hg37,
    prot_seq_df_hg37
)

annot_df_hg38 = merge_uniprot_protein_seq(
    uniprot_df_hg38,
    prot_seq_df_hg38
)


##3. MERGE the annot_df with the clinvar files
def merge_variants_annotations(variants_df, annot_df):

    # Preserve a unique ID for every original variant row
    variants_df = variants_df.copy()
    variants_df["_variant_row"] = range(len(variants_df))

    # Merge annotations
    merged_df = variants_df.merge(
        annot_df,
        left_on=["Gene", "Feature"],
        right_on=["gene_stable_id", "transcript_stable_id"],
        how="left"
    )

    # Original variant rows that received at least one annotation
    mapped_ids = merged_df.loc[
        merged_df["protein_stable_id"].notna(),
        "_variant_row"
    ].unique()

    n_total = len(variants_df)
    n_mapped = len(mapped_ids)
    n_unmapped = n_total - n_mapped

    print(f"Starting variants: {n_total:,}")
    print(f"Mapped variants:   {n_mapped:,} ({n_mapped/n_total:.2%})")
    print(f"Unmapped variants: {n_unmapped:,} ({n_unmapped/n_total:.2%})")
    
    
    unmapped = variants_df.loc[
    ~variants_df.index.isin(
        merged_df.loc[
            merged_df["protein_stable_id"].notna(),
            "_variant_row"
        ]
    )
]
    print(f"unmapped variants: {unmapped}")

    return merged_df


annotated_variants_hg37 = merge_variants_annotations(
    variants_hg37,
    annot_df_hg37
)

annotated_variants_hg37.to_csv(os.path.join(mount_results, "annotated_variants_hg37_for_vep_training_filter.tsv"), sep="\t", index=False)

annotated_variants_hg38 = merge_variants_annotations(
    variants_hg38,
    annot_df_hg38
)

annotated_variants_hg38.to_csv(os.path.join(mount_results, "annotated_variants_hg38_for_vep_training_filter.tsv"), sep="\t", index=False)






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
    
    step_dataframes = {

            "vep_out_hg38_df": vep_out_hg38_df,
            "vep_out_hg37_df": vep_out_hg37_df,
            "MANE_Select_hg38": mane_select_transcripts_hg38,
            "Canonical_hg37": canonical_transcripts_hg37,
            "missense_hg38": vep_missense_hg38,
            "missense_hg37": vep_missense_hg37,
            "AF_hg38": AF_filtered_hg38,
            "AF_hg37": AF_filtered_hg37}

    df_summary = build_filter_summary_table(step_dataframes)

    display(df_summary)

    plot_filter_summary(df_summary, metric="UniqueVariationID")
    plot_filter_summary(df_summary, metric="UniqueGeneSymbol")


    
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
    
    
    


    
    
    
    
    