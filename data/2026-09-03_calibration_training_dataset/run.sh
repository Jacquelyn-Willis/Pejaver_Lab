#!/bin/bash
#BSUB -P acc_oscarlr         
#BSUB -q interactive              
#BSUB -n 1                      # one core is plenty for downloading
#BSUB -W 12:00                    
#BSUB -J calibration              # descriptive job name
#BSUB -o /sc/arion/projects/pejaverlab/users/willij115/logs/calibration.%J.out.txt       # stdout 
#BSUB -e /sc/arion/projects/pejaverlab/users/willij115/logs/calibration.%J.err.txt      # stderr 
#BSUB -Is /bin/bash



#to cause any big bugs to fail out of the script
set -eou pipefail 
set -x



data=/sc/arion/projects/pejaverlab/users/willij115/data/2026-09-03_calibration_training_dataset 
results=/sc/arion/projects/pejaverlab/users/willij115/results/2026-09-03_calibration_training_dataset

# --- Get the ClinVar test/calibration data ---

get_clinvar_test_data () {

    cd $data || exit 1. #do this command or exit if it fails

    #wget -c https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/2019/variant_summary_2019-12.txt.gz 

    wget -c https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/2024/variant_summary_2024-12.txt.gz 
}



get_clinvar_test_data

get_mp2_training_data () { 

    #copy from local machine to HPC, Yile sent to me on slack 
    cp /Users/jwillis/Downloads/mp2_actual_training_data.txt "/Users/jwillis/minerva/pejaverlab/data/2026-09-03_calibration_training_dataset"

}


get_polyphen2_training_data () {

    cd $data || exit 1. #do this command or exit if it fails

    wget -c https://genetics.bwh.harvard.edu/downloads/pph2/training/training-2.2.2.tar.gz


}



get_db () {

    cd $data || exit 1. #do this command or exit if it fails

    
}


get_final_mmmc2_final_df_for_concordance () {

    cd $data || exit 1. #do this command or exit if it fails

    cp  /Users/jwillis/Downloads/mmc2-3.xlsx    /Users/jwillis/minerva/pejaverlab/data/2026-09-03_calibration_training_dataset
}

get_uniprotID_and_prot_seq_from_ensembl_ftp () {

     cd $data || exit 1. #do this command or exit if it fails

     wget -c https://ftp.ensembl.org/pub/current/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz
	
     wget -c https://ftp.ensembl.org/pub/current/tsv/homo_sapiens/Homo_sapiens.GRCh38.116.uniprot.tsv.gz

     wget -c https://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/pep/Homo_sapiens.GRCh37.pep.all.fa.gz

     wget -c https://ftp.ensembl.org/pub/grch37/current/tsv/homo_sapiens/Homo_sapiens.GRCh37.85.uniprot.tsv.gz
}


#fucnction calls 

#get_mp2_training_data
#get_clinvar_test_data
#get_polyphen2_training_data
get_uniprotID_and_prot_seq_from_ensembl_ftp
 
