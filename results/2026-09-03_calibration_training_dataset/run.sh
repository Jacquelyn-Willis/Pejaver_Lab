#!/bin/bash
#BSUB -J ensembl_vep_hg37              # Job name
#BSUB -P acc_pejaverlab                    # Project allocation
#BSUB -q express                       # Queue name
#BSUB -n 8                              # 8 compute cores
#BSUB -R "rusage[mem=10000]"             # 10 GB per core → 80 GB total
#BSUB -R "span[hosts=1]"                # All cores on the same node
#BSUB -W 12:00                          # 85 hour wall-time limit
#BSUB -o /sc/arion/projects/pejaverlab/users/willij115/results/2026-09-03_calibration_training_dataset/err_files/clinvar_processing_hg37.%J.out.txt       # STDOUT log
#BSUB -eo /sc/arion/projects/pejaverlab/users/willij115/results/2026-09-03_calibration_training_dataset/err_files/clinvar_processing_hg37.%J.err.txt      # STDERR log
#BSUB -L /bin/bash


#to cause any big bugs to fail out of the script
set -eou pipefail 
set -x


# --- Conda ---
source "/hpc/packages/minerva-rocky9/miniforge3/26.1.1-3/miniforge/etc/profile.d/conda.sh"
module load anaconda3/latest

module load vep/113
module load python



# Directories
error_files="/sc/arion/projects/pejaverlab/users/willij115/results/2026-09-03_calibration_training_dataset/err_files"
data="/sc/arion/projects/pejaverlab/users/willij115/data/2026-09-03_calibration_training_dataset" 
results="/sc/arion/projects/pejaverlab/users/willij115/results/2026-09-03_calibration_training_dataset"




#call python script to filter clinvar dataset for ensembl VEP 
function filter_clinvar_data_for_vep (){

    python3 2025-09-04-filter_clinvar_dataset_for_ensemblVEP.py
}


run_vep_on_semi_filtered_clinvar_data_hg38 (){

    mkdir -p ${results}/vep_output_hg38

    sort -k1,1V -k2,2n clinvar_filtered_for_ensemblVEP_hg38.tsv > clinvar_filtered_for_ensemblVEP_hg38_sorted.tsv

    vep \
    -i ${results}/clinvar_filtered_for_ensemblVEP_hg38_sorted.tsv \
    -o ${results}/clinvar_filtered_for_ensemblVEP_hg38_vep_output.txt \
    --offline \
    --cache \
    --uniprot \
    --format ensembl \
    --dir_cache "$pathToCache" \
    --assembly GRCh38 \
    --tab \
    --symbol --biotype --canonical --mane_select --tsl \
    --af_gnomade --af_gnomadg \
    --pick --pick_order mane_select,canonical,rank,length \
    --fork 7 \
    --force_overwrite

}

run_vep_on_semi_filtered_clinvar_data_hg37 (){

    mkdir -p ${results}/vep_output_hg37

    sort -k1,1V -k2,2n clinvar_filtered_for_ensemblVEP_hg37.tsv > clinvar_filtered_for_ensemblVEP_hg37_sorted.tsv

    vep \
    -i ${results}/clinvar_filtered_for_ensemblVEP_hg37_sorted.tsv \
    -o ${results}/clinvar_filtered_for_ensemblVEP_hg37_vep_output.txt \
    --offline \
    --cache \
    --uniprot \
    --format ensembl \
    --dir_cache "$pathToCache" \
    --assembly GRCh37 \
    --tab \
    --symbol --biotype --canonical \
     --af_gnomade --af_gnomadg \
    --pick --pick_order canonical,rank,length \
    --fork 7 \
    --force_overwrite

}

#run_vep_on_semi_filtered_clinvar_data_hg38
#run_vep_on_semi_filtered_clinvar_data_hg37
#filter_clinvar_data_for_vep 

run_mutpred_filter() {

    python3 2025-09-15-filter_clinvar_dataset_for_mutpred2_training_variants.py


    }

#run_mutpred_filter

run_polyphen_filter() {

    python3 2025-09-15-filter_clinvar_dataset_for_polyphen2_training_variants.py


    }
run_polyphen_filter