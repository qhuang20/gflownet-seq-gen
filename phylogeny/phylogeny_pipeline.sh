# MUSCLE
muscle -align LET7_family_mature_ALLspecies.fa -output LET7_family_mature_ALLspecies.aln
# TRIMAL
trimal -in LET7_family_mature_ALLspecies.aln -out LET7_family_mature_ALLspecies.clustal -clustal
# IQTREE
iqtree -s LET7_family_mature_ALLspecies.clustal  --seed 42  -T AUTO -m MFP -B 6000 --ancestral --sup-min 0.95 --prefix miRNA_iqtree/miRNA_LET7