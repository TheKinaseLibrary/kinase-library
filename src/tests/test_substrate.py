import kinase_library as kl


def test_substrate_score():
    sub = kl.Substrate('PSVEPPLtQETFSDL')
    score_dict = sub.score(kinases=['AKT1', 'AKT2'], output_type='dict')
    assert score_dict == {'AKT1': -6.2244, 'AKT2': -6.3716}


def test_substrate_percentile():
    sub = kl.Substrate('PSVEPPLtQETFSDL')
    percentile_dict = sub.percentile(kinases=['AKT1', 'AKT2'], output_type='dict')
    assert percentile_dict == {'AKT1': 11.94, 'AKT2': 3.2}


def test_substrate_score_rank():
    sub = kl.Substrate('PSVEPPLtQETFSDL')
    score_rank_dict = sub.rank(method="score", kinases=['AKT1', 'AKT2'], output_type='dict')
    assert score_rank_dict == {'AKT1': 288, 'AKT2': 291}


def test_substrate_percentile_rank():
    sub = kl.Substrate('PSVEPPLtQETFSDL')
    percentile_rank_dict = sub.rank(method="percentile", kinases=['AKT1', 'AKT2'], output_type='dict')
    assert percentile_rank_dict == {'AKT1': 268, 'AKT2': 307}
