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

#LOAD DATA

input_hg38 = pd.read_csv(os.path.join(mount_results, "clinvar_hg38_post_ensemble_vep_filters_w_uniprot.tsv"),
        sep="\t", header = 0)

input_hg37 = pd.read_csv(os.path.join(mount_results, "clinvar_hg37_post_ensemble_vep_filters_w_uniprot.tsv"),
        sep="\t", header = 0)


#METHOD1:
##1. use uniprot ID from ensembl VEP output
##2. map them to polyphen2 uniporot ID and AA change and pos

#METHOD2:
##1. pull down uniprot ID from ensembl 
##2. map them to polyphen2 uniporot ID and AA change and pos

##POST ANALYSIS
##1. compare concordanance of methods 
#### which method allowed more accurate mapping ? 



#METHOD1: remove polyphen training variants that overlap 

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


def remove_polyphen_variants(final_df, polyphen_train):

    # Clean ClinVar's UniProt ID (SwissProt first, fallback to TREMBL)
    def extract_uniprot_id(value):
        if pd.isna(value) or value == '-':
            return None
        first = value.split(',')[0].split(';')[0].strip()
        return first.split('.')[0]

    final_df['UNIPROT_ID'] = final_df['SWISSPROT'].apply(extract_uniprot_id)
    final_df['UNIPROT_ID'] = final_df['UNIPROT_ID'].fillna(
        final_df['TREMBL'].apply(extract_uniprot_id)
    )

    # Split Amino_acids column ("S/N") into WT/Mut
    aa_split = final_df['Amino_acids'].str.split('/', expand=True)
    final_df['WT_AA'] = aa_split[0]
    final_df['Mut_AA'] = aa_split[1]

    # Build matching key on ClinVar side
    final_df['match_key'] = (
        final_df['UNIPROT_ID'].astype(str) + "_" +
        final_df['Protein_position'].astype(str) + "_" +
        final_df['WT_AA'].astype(str) + "_" +
        final_df['Mut_AA'].astype(str)
    )

    # Build matching key on PolyPhen training side
    polyphen_train['match_key'] = (
        polyphen_train['uniprot_id'].astype(str) + "_" +
        polyphen_train['position'].astype(str) + "_" +
        polyphen_train['ref_aa'].astype(str) + "_" +
        polyphen_train['alt_aa'].astype(str)
    )

    polyphen_keys = set(polyphen_train['match_key'])

    print(f"gene count Before filtering: {final_df['Gene'].nunique()}")
    print(f"variant count Before filtering: {final_df['Uploaded_variation'].nunique()}")
    final_df_filtered = final_df[~final_df['match_key'].isin(polyphen_keys)].copy()
    print(f"gene count After filtering: {final_df_filtered['Gene'].nunique()}")
    print(f"variant count Before filtering: {final_df_filtered['Uploaded_variation'].nunique()}")
    print(f"genes Removed: {final_df['Gene'].nunique() - final_df_filtered['Gene'].nunique()}")
    print(f"variants Removed: {final_df['Uploaded_variation'].nunique() - final_df_filtered['Uploaded_variation'].nunique()}")
    

    final_df_filtered = final_df_filtered.drop(columns=['match_key', 'WT_AA', 'Mut_AA'])

    return final_df_filtered


final_df_filtered_hg38 = remove_polyphen_variants(input_hg38, poly_phen_training_set)
final_df_filtered_hg38.to_csv(os.path.join(mount_results, "clivar_filtered_for_polyphen_hg38_method1.tsv"), sep="\t", index=False)

final_df_filtered_hg37 = remove_polyphen_variants(input_hg37, poly_phen_training_set)
final_df_filtered_hg37.to_csv(os.path.join(mount_results, "clivar_filtered_for_polyphen_hg37_method1.tsv"), sep="\t", index=False)


#METHOD2:


#1. get uniprot ID from ensemble rest

def get_uniprot_id_hg38(gene_id):
    url = f"https://rest.ensembl.org/xrefs/id/{gene_id}"

    r = requests.get(
        url,
        params={"external_db": "Uniprot/SWISSPROT"},
        headers={"Content-Type": "application/json"}
    )

    if r.status_code != 200:
        return None

    results = r.json()
    if len(results) == 0:
        return None

    return results[0]["primary_id"]


def add_uniprot_ids_hg38(final_df):
    ensembl_gene = final_df["Gene"].dropna().unique()
    print(len(ensembl_gene))
    print(ensembl_gene[:10])

    uniprot_map_hg38 = {}
    for gene1 in ensembl_gene:
        uniprot_map_hg38[gene1] = get_uniprot_id_hg38(gene1)
        time.sleep(0.1)

    final_df["Ensembl_UniProt_ID"] = final_df["Gene"].map(uniprot_map_hg38)
    return final_df


def get_uniprot_id_hg37(gene_id):
    url = f"https://grch37.rest.ensembl.org/xrefs/id/{gene_id}"

    r = requests.get(
        url,
        params={"external_db": "Uniprot/SWISSPROT"},
        headers={"Content-Type": "application/json"}
    )

    if r.status_code != 200:
        return None

    results = r.json()
    if len(results) == 0:
        return None

    return results[0]["primary_id"]


def add_uniprot_ids_hg37(final_df):
    ensembl_gene = final_df["Gene"].dropna().unique()
    print(len(ensembl_gene))
    print(ensembl_gene[:10])

    uniprot_map_hg37 = {}
    for gene1 in ensembl_gene:
        uniprot_map_hg37[gene1] = get_uniprot_id_hg37(gene1)
        time.sleep(0.1)

    final_df["Ensembl_UniProt_ID"] = final_df["Gene"].map(uniprot_map_hg37)
    return final_df


#2. merge and remove overlap

def remove_polyphen_variants(final_df, polyphen_train):

    # Split Amino_acids column ("S/N") into WT/Mut
    aa_split = final_df['Amino_acids'].str.split('/', expand=True)
    final_df['WT_AA'] = aa_split[0]
    final_df['Mut_AA'] = aa_split[1]

    # Build matching key on ClinVar side using Ensembl-derived UniProt ID
    final_df['match_key'] = (
        final_df['Ensembl_UniProt_ID'].astype(str) + "_" +
        final_df['Protein_position'].astype(str) + "_" +
        final_df['WT_AA'].astype(str) + "_" +
        final_df['Mut_AA'].astype(str)
    )

    # Build matching key on PolyPhen training side
    polyphen_train['match_key'] = (
        polyphen_train['uniprot_id'].astype(str) + "_" +
        polyphen_train['position'].astype(str) + "_" +
        polyphen_train['ref_aa'].astype(str) + "_" +
        polyphen_train['alt_aa'].astype(str)
    )

    polyphen_keys = set(polyphen_train['match_key'])

    print(f"gene count Before filtering: {final_df['Gene'].nunique()}")
    print(f"variant count Before filtering: {final_df['Uploaded_variation'].nunique()}")

    final_df_filtered = final_df[~final_df['match_key'].isin(polyphen_keys)].copy()

    print(f"gene count After filtering: {final_df_filtered['Gene'].nunique()}")
    print(f"variant count After filtering: {final_df_filtered['Uploaded_variation'].nunique()}")
    print(f"genes Removed: {final_df['Gene'].nunique() - final_df_filtered['Gene'].nunique()}")
    print(f"variants Removed: {final_df['Uploaded_variation'].nunique() - final_df_filtered['Uploaded_variation'].nunique()}")

    final_df_filtered = final_df_filtered.drop(columns=['match_key', 'WT_AA', 'Mut_AA'])

    return final_df_filtered



final_df_filtered_hg38 = remove_polyphen_variants(input_hg38, poly_phen_training_set)
final_df_filtered_hg38.to_csv(
    os.path.join(mount_results, "clinvar_filtered_for_polyphen_hg38_method2.tsv"),
    sep="\t", index=False
)

final_df_filtered_hg37 = remove_polyphen_variants(input_hg37, poly_phen_training_set)
final_df_filtered_hg37.to_csv(
    os.path.join(mount_results, "clinvar_filtered_for_polyphen_hg37_method2.tsv"),
    sep="\t", index=False
)

















'''


def get_uniprot_id_hg38(ensembl_id):
  url = f"https://rest.ensembl.org/xrefs/id/{ensembl_id}"

  r = requests.get(
      url,
      params={
          "external_db": "UniProtKB/Swiss-Prot"
      },  # Use "UniProtKB/Swiss-Prot" for reviewed proteins (recommended)
      headers={"Content-Type": "application/json"},
  )

  if r.status_code != 200:
    return None

  results = r.json()

  if not results or len(results) == 0:
    return None

  return results[0]["primary_id"]


def get_uniprot_id_hg37(ensembl_gene):
  url = f"https://grch37.rest.ensembl.org/xrefs/id/{ensembl_gene}"

  r = requests.get(
      url,
      params={"external_db": "UniProtKB/Swiss-Prot"},
      headers={"Content-Type": "application/json"},
  )

  if r.status_code != 200:
    return None

  results = r.json()

  if not results:
    return None

  return results[0]["primary_id"]




#usage:
 
poly_phen_training_set = load_poly_phen_data()

ensembl_genes_hg38 = get_ensembl_gene(final_hg37)
ensembl_genes_hg37 = get_ensembl_gene(final_hg38)


uniprot_map_hg38 = {}
for gene1 in ensembl_genes_hg38:
    uniprot_map_hg38[gene1] = uniprot_map_hg38[gene1]
    
final_hg38["Uniprot_ID"] = final_hg38["Gene"].map(uniprot_map_hg38)




uniprot_map_hg37 = {}
for gene2 in ensembl_genes_hg37:
    uniprot_map_hg37[gene2] = uniprot_map_hg37[gene2]

final_hg37["Entrez_ID"] = final_hg37["Gene"].map(uniprot_map_hg37)

#save output

final_hg38.to_csv(
        os.path.join(mount_results, "clinvar_polyphen_hg38.tsv"),
        sep="\t",
        index=False
    )

final_hg37.to_csv(
        os.path.join(mount_results, "clinvar_removed_polyphen_hg37.tsv"),
        sep="\t",
        index=False
    )



#REmove polyphen2 training variants also




'''