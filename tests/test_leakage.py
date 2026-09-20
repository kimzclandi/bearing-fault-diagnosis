import pytest
from bearing_diagnosis.data import manifest,check_duplicates
from bearing_diagnosis.config import load_config

def test_disjoint_conditions_and_matching_configs():
    quick=manifest(load_config('configs/quick.yaml'));full=manifest(load_config('configs/full.yaml'))
    assert quick.equals(full);assert len(quick)==40
    groups=[set(quick[quick.split==s].source_id) for s in ['train','val','test']]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert set(quick[quick.split=='test'].load_hp)=={3}

def test_duplicate_signal_under_new_filename_rejected():
    records=[{'source_id':'a','sha256':'a','signal_sha256':'same'}, {'source_id':'b','sha256':'b','signal_sha256':'same'}]
    with pytest.raises(ValueError,match='Duplicate'):check_duplicates(records)


def test_source_99_selects_own_channel_not_embedded_source_98():
    import numpy as np
    from bearing_diagnosis.data import select_channel
    mat={'X098_DE_time':np.zeros(200),'X099_DE_time':np.ones(200),'X098RPM':1772}
    key,rpm,ignored=select_channel(mat,'99')
    assert key=='X099_DE_time' and rpm==[] and ignored==['X098_DE_time']


def test_manifest_optional_fields_are_json_serializable():
    import json
    table=manifest(load_config('configs/quick.yaml'))
    json.dumps(table.to_dict('records'),allow_nan=False)
    assert table.loc[table.label==0,'outer_position'].eq('').all()
