# Cluster-based cross-validation splits


Steps 1 and 2 follow the clustering utilities of
[CAMP](https://github.com/twopin/CAMP/tree/master/cluster), ported to Python 3.
Steps 3 and 4 are this project's own.

The splits themselves are distributed under `data/`, so this pipeline only needs to be
run to rebuild them or to change the threshold.

## Step 1 — alignment scores

Write the FASTA files from the dataset:

    python 1_build_fasta.py



Then align each file against itself with the
[Complete Striped Smith-Waterman Library](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library),
which is not included here and has to be obtained and built separately. With its
`src` directory as `smith-waterman-src/`:

    cd smith-waterman-src/
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$PWD
    python pyssw.py -p ../fasta/proteins.fasta ../fasta/proteins.fasta > proteins_output.txt
    python pyssw.py -p ../fasta/peptides.fasta ../fasta/peptides.fasta > peptides_output.txt



## Step 2 — similarity matrices

Raw alignment scores are normalised so that sequences of different lengths are
comparable and a sequence scores 1 against itself:

    similarity(i, j) = SW(i, j) / sqrt( SW(i, i) * SW(j, j) )

    python 2_sequence_similarity.py --fasta fasta/proteins.fasta \
                                    --alignment smith-waterman-src/proteins_output.txt \
                                    --out sim_mat_protein.npy
    python 2_sequence_similarity.py --fasta fasta/peptides.fasta \
                                    --alignment smith-waterman-src/peptides_output.txt \
                                    --out sim_mat_peptide.npy

## Step 3 — clustering

Average-linkage hierarchical clustering on the distance matrix `1 - similarity`, cut at
the threshold:

    python 3_cluster_sequences.py --fasta fasta/proteins.fasta \
                                  --sim-mat sim_mat_protein.npy --prefix prot
    python 3_cluster_sequences.py --fasta fasta/peptides.fasta \
                                  --sim-mat sim_mat_peptide.npy --prefix pep

Writes `similarity_cluster_dict/{prot,pep}_cluster_dict_<threshold>.pkl`, mapping each
cluster label to its member chains. Thresholds 0.3, 0.4, 0.5 and 0.6 are produced by
default; 0.5 is the setting reported in the paper.

## Step 4 — cross-validation splits

Folds are drawn over clusters, and ten percent of each fold's training clusters are set
aside for validation:

    python 4_split_data.py --setting protein --threshold 0.5    # new-protein evaluation
    python 4_split_data.py --setting peptide --threshold 0.5    # new-peptide evaluation
    python 4_split_data.py --setting pair    --threshold 0.5    # new-pair evaluation


Because clusters differ in size, the pair counts vary from fold to fold even though the
cluster counts do not. The default seeds were chosen to keep the ratio near 8:1:1; to
try others without writing anything:

    python 4_split_data.py --setting protein --dry-run --seeds 7 
    python 4_split_data.py --setting pair    --dry-run --seeds 7 


