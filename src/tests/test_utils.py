import kinase_library as kl


def test_sequence_to_substrate():
    sequences = ["PPLs*",
                 "PLs*QE",
                 "SVEPPLs*QEtFSD"]
    substrates = []
    for sequence in sequences:
        asterisk_index = sequence.index("*")
        # Remove asterisk
        sequence = sequence.replace('*', '')
        substrate = kl.sequence_to_substrate(sequence, phos_pos=asterisk_index)
        substrates.append(substrate)
    assert substrates == ['____PPLs_______',
                          '_____PLsQE_____',
                          '_SVEPPLsQETFSD_']
