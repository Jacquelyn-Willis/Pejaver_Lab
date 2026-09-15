import sys
import pandas as pd 
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import requests

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





### PLOT and summary

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
                    "UniqueGeneSymbol": asm_df["Gene"].nunique(),
                    "UniqueVariationID": asm_df["Uploaded_variation"].nunique(),
                })
        else:
            # Fallback if a step DataFrame lacks an Assembly column
            rows.append({
                "FilterStep": step_label,
                "Assembly": "All",
                "UniqueGeneSymbol": step_df["Gene"].nunique(),
                "UniqueVariationID": step_df["Uploaded_variation"].nunique(),
            })
    return pd.DataFrame(rows)


def build_filter_summary_table(
    step_dataframes,
    assembly_col="Assembly",
    assemblies=("GRCh37", "GRCh38")
):
    rows = []

    for step_label, step_df in step_dataframes.items():
        if assembly_col in step_df.columns:
            for asm in assemblies:
                asm_df = step_df[step_df[assembly_col] == asm]

                rows.append({
                    "FilterStep": step_label,
                    "Assembly": asm,
                    "UniqueGeneSymbol": asm_df["Gene"].nunique(),
                    "UniqueVariationID": asm_df["Uploaded_variation"].nunique(),
                })
        else:
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


def get_ensembl_entrez_ID(af_df) :

    # Get unique Ensembl gene IDs
    ensembl_genes = af_df["Gene"].dropna().unique()

    print(len(ensembl_genes))
    print(ensembl_genes[:10])
    
    return ensembl_genes




def get_entrez_id_hg38(ensembl_gene):
    url = f"https://rest.ensembl.org/xrefs/id/{ensembl_gene}"

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


def get_entrez_id_hg37(ensembl_gene):
    url = f"https://grch37.rest.ensembl.org/xrefs/id/{ensembl_gene}"

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

#usager: 

ensembl_genes_hg38 = get_ensembl_entrez_ID(AF_filtered_hg38)
ensembl_genes_hg37 = get_ensembl_entrez_ID(AF_filtered_hg37)

entrez_map_hg38 = {}
for gene1 in ensembl_genes_hg38:
    entrez_map_hg38[gene1] = get_entrez_id_hg38(gene1)
    
final_hg38["Entrez_ID"] = final_hg38["Gene"].map(entrez_map_hg38)


entrez_map_hg37 = {}
for gene2 in ensembl_genes_hg37:
    entrez_map_hg37[gene2] = get_entrez_id_hg37(gene2)

final_hg37["Entrez_ID"] = final_hg37["Gene"].map(entrez_map_hg37)



###protein to variant annotation

# Keep missense variants only
def convert_to_protein_variant_annot(final_df):
    # Split S/N into reference and alternate amino acids
    final_df[["Ref_AA", "Alt_AA"]] = final_df["Amino_acids"].str.split(
        "/", expand=True
    )

    # Construct training-style protein variant: S21N
    final_df["protein_variant"] = (
        final_df["Ref_AA"]
        + final_df["Protein_position"].astype(int).astype(str)
        + final_df["Alt_AA"]
    )

    # Check result

    final_df.head()
    return final_df


mutpred_to_merge_and_remove_hg38 = convert_to_protein_variant_annot(final_hg38)
mutpred_to_merge_and_remove_hg37 = convert_to_protein_variant_annot(final_hg37)



#REmove polyphen2 training variants also












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
    
    
    


    
    
    
    
    