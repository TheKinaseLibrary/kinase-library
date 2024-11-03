"""
####################################
# The Kinase Library - Global Data #
####################################
"""

# Matrices
_default_mat_dir = './../databases/matrices'
def reset_mat_dir():
    global mat_dir
    mat_dir = _default_mat_dir
reset_mat_dir()

# Phosphoproteome
_default_phosprot_name = 'ochoa'
def reset_phosprot_name():
    global phosprot_name
    phosprot_name = _default_phosprot_name
reset_phosprot_name()
phosprot_file_type = ['parquet','txt','tsv','csv','xls','xlsx']
phosprot_path='./../databases/substrates'
phosprot_info_columns = {'date_added': 'Date added (yyyy-mm-dd)', 'date_updated': 'Date updated (yyyy-mm-dd)', 'name': 'Name', 'description': 'Description', 'ser_thr_sites': 'Ser-Thr sites', 'tyrosine_sites': 'Tyr sites'}
# global all_scored_phosprot

# Amino acids
aa_unmod = ['P','G','A','C','S','T','V','I','L','M','F','Y','W','H','K','R','Q','N','D','E']
aa_phos = ['P','G','A','C','S','T','V','I','L','M','F','Y','W','H','K','R','Q','N','D','E','s','t','y']
aa_all_ptm = ['P','G','A','C','S','T','V','I','L','M','F','Y','W','H','K','R','Q','N','D','E','s','t','y','kac','kmet']

# Random amino acids
ser_thr_random_aa = ['P','G','A','V','I','L','M','F','Y','W','H','K','R','Q','N','D','E']
tyrosine_random_aa = ['P','G','A','S','T','V','I','L','M','F','W','H','K','R','Q','N','D','E']
random_aa_value = {'ser_thr': 1/17, 'tyr': 1/18, 'ser_thr_tyr': 1/16}

# Positions
ser_thr_pos = [-5,-4,-3,-2,-1,1,2,3,4]
tyrosine_pos = [-5,-4,-3,-2,-1,1,2,3,4,5]

# Phosphoacceptor
kin_type_phos_acc = {'ser_thr': ['S','T','s','t'], 'tyrosine': ['Y','y']}

# Valid arguments
valid_kin_types = {'ser_thr','tyrosine'}
valid_phos_res = {'S','s','T','t','Y','y'}
valid_mat_type = {'densitometry','raw','norm','log2'}
valid_mat_scale = {'linear','log'}
valid_name_types = {'kinase','gene','matrix','uniprot_id'}
valid_aa = {'P','G','A','C','S','T','V','I','L','M','F','Y','W','H','K','R','Q','N','D','E','s','t','y','X','x','-','_','*'}
valid_score_output_type = {'series','list','dict'}
valid_output_sort_by= {'input','name','value','score','percentile'}

# Valid scoring and enrichment parameters
valid_scoring_metric = {'score','percentile'}
valid_enrichment_data_type = {'foreground','fg','background','bg'}
valid_dp_enrichment_type = {'upregulated','downregulated','combined'}
valid_dp_sites_type = {'upregulated','upreg','downregulated','downreg','unregulated','unreg'}
valid_kl_method = {'score','percentile','score_rank','percentile_rank'}
valid_score_type = {'scores','score_ranks','percentiles','percentile_ranks'}
valid_enrichment_type = {'enriched','depleted','both'}

# Plotting parameters
valid_color_kins_method = {'family', 'type'}
type_colors = {'ser_thr': '#ff0000', 'tyrosine': '#008000'}
family_colors = {'AGC': '#ff0000', 'Alpha': '#d19662', 'CAMK': '#8e44ad', 'CK1': '#8f2323', 'CMGC': '#008000', 'FAM20': '#ff1493', 'Other': '#8b4513', 'PDHK': '#ff7f00', 'PIKK': '#4b0082', 'STE': '#8f6a23', 'TKL': '#0000ff', 'ABL': '#808080', 'ACK': '#228b22', 'CSK': '#800000', 'DDR': '#ffa500', 'EPHR': '#808000', 'ErbB': '#483d8b', 'FAK': '#9acd32', 'FES': '#00008b', 'FGFR': '#8fbc8f', 'FRK': '#800080', 'HGFR': '#b03060', 'INSR': '#ff4500', 'JAK': '#8a2be2', 'LTKR': '#ffff00', 'MuSK': '#ff8c00', 'NGFR': '#00fa9a', 'PDGFR': '#ffb6c1', 'RETR': '#fa8072', 'ROSR': '#191970', 'SRC': '#da70d6', 'SYK': '#f0e68c', 'TAMR': '#00bfff', 'TEC': '#ff00ff', 'TIER': '#008080', 'TNNI3K': '#1e90ff', 'VEGFR': '#ffc0cb', 'WEE': '#add8e6'}
valid_cluster_method = {'lff', 'pval', 'both', 'custom'}
aa_colors = {'D': '#DC143C', 'E': '#DC143C', 's': '#DC143C', 't': '#DC143C', 'y': '#DC143C', 'pS': '#DC143C', 'pT': '#DC143C', 'pS/pT': '#DC143C', 'pY': '#DC143C', 'R': '#0000FF', 'K': '#0000FF', 'C': '#DAA520', 'F': '#DAA520', 'Y': '#DAA520', 'W': '#DAA520', 'V': '#DAA520', 'I': '#DAA520', 'L': '#DAA520', 'M': '#DAA520', 'Q': '#8A2BE2', 'N': '#8A2BE2', 'H': '#8A2BE2', 'S': '#8A2BE2', 'T': '#8A2BE2', 'A': '#008000', 'G': '#008000', 'P': '#000000'}
# aa_colors = {'D': '#DC143C', 'E': '#DC143C', 's': '#DC143C', 't': '#DC143C', 'y': '#DC143C', 'pS': '#DC143C', 'pT': '#DC143C', 'pS/pT': '#DC143C', 'pY': '#DC143C', 'R': '#0000FF', 'K': '#0000FF', 'C': '#DAA520', 'F': '#946000', 'Y': '#946000', 'W': '#946000', 'V': '#F4C000', 'I': '#F4C000', 'L': '#F4C000', 'M': '#F4C000', 'Q': '#8A2BE2', 'N': '#8A2BE2', 'H': '#8A2BE2', 'S': '#8A2BE2', 'T': '#8A2BE2', 'A': '#008000', 'G': '#008000', 'P': '#000000'}

# Others
kl_method_comp_direction_dict = {'score':'higher', 'percentile':'higher', 'score_rank':'lower', 'percentile_rank':'lower'}
default_seq_col = 'Sequence'