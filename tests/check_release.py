"""Public release checks; no access to the private training dataset is needed."""
from pathlib import Path
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    forbidden = {'training_database.csv','updated_older_DB.xlsx','original_split_membership.csv',
                 'record_inventory.csv','SHAP_values.csv'}
    for folder in ['data','models','config','examples','results']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file():
                assert path.name not in forbidden, path
                assert path.stat().st_size < 100_000_000, path
    sites = pd.read_csv(ROOT/'data/sites/Lisbon_Vs30_prediction_grid.csv')
    assert len(sites)==1375 and sites.grid_id.is_unique
    assert list(sites.columns)==['grid_id','station_latitude','station_longitude','vs30_m_s']
    assert np.isfinite(sites).all().all() and (sites.vs30_m_s>0).all()
    full = pd.read_csv(ROOT/'data/sites/Original_full_scenario_grid.csv')
    assert len(full)==2652 and full.grid_id.is_unique
    assert sites.grid_id.isin(full.grid_id).all()
    source = full.set_index('grid_id').loc[sites.grid_id]
    assert np.allclose(sites.iloc[:,1:],source.iloc[:,0:],atol=1e-12,rtol=0)
    with zipfile.ZipFile(ROOT/'data/sites/Lisbon_Vs30_prediction_grid.xlsx') as z:
        ns = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        workbook = ET.fromstring(z.read('xl/workbook.xml'))
        assert len(workbook.findall('m:sheets/m:sheet',ns))==1
        rows = ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('m:sheetData/m:row',ns)
        assert len(rows)==1376
        for row,expected in zip(rows[1:],sites.to_numpy()):
            actual = [float(cell.find('m:v',ns).text) for cell in row.findall('m:c',ns)]
            assert np.allclose(actual,expected,atol=1e-12,rtol=0)
    for scenario in 'ACEFGH':
        d = pd.read_csv(ROOT/f'results/scenarios/{scenario}_Lisbon_centres_XGB29_GMPE_levels.csv')
        assert len(d)==1375 and set(d.SITE_ID)==set(sites.grid_id)
        assert d.CENTRE_IN_LISBON.all()
    partition = pd.read_csv(ROOT/'config/record_partition.csv')
    assert len(partition)==9579 and partition.record_id.tolist()==list(range(1,9580))
    assert partition.split.eq('SCR_validation').sum()==1343
    assert partition.analysis_group.eq('SCR').sum()==4479
    assert set(partition.columns)=={'record_id','station_setting_original','analysis_group','split',
                                    'training_position','bootstrap_event_group'}
    real = pd.read_csv(ROOT/'results/tables/real_recordings_station_predictions.csv')
    table = pd.read_csv(ROOT/'results/tables/real_recordings_available_and_common_subset_metrics.csv')
    common = real[['XGB29','AB06','ABMIX','FIVE','KOTHA']].notna().all(axis=1)
    assert len(real)==10 and common.sum()==9 and real.loc[~common,'station'].tolist()==['IV.MGR']
    for row in table.itertuples():
        cohort = real if row.scope=='all_available_10_article_recordings' else real[common]
        cohort = cohort[cohort[row.model_code].notna()]
        error = cohort[row.model_code]-cohort.observed_pga_pctg
        assert len(cohort)==row.n
        assert math.isclose(error.abs().mean(),row.MAE_pctg,abs_tol=1e-10)
        assert math.isclose(np.sqrt((error**2).mean()),row.RMSE_pctg,abs_tol=1e-10)
    mc = pd.read_csv(ROOT/'results/tables/paired_MC50_repetition_metrics.csv')
    summary = pd.read_csv(ROOT/'results/tables/paired_MC50_summary.csv')
    assert len(mc)==100
    for row in summary.itertuples():
        values = mc.loc[mc.protocol.eq(row.protocol),row.metric]
        assert len(values)==50 and math.isclose(values.mean(),row.mean,abs_tol=1e-10)
        assert math.isclose(values.std(),row.sd,abs_tol=1e-10)
    manifest = json.loads((ROOT/'results/asset_provenance.json').read_text())
    for entry in manifest:
        assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
    verification = json.loads((ROOT/'models/inference_export_verification.json').read_text())
    for value in verification.values():
        assert value['prediction_max_difference_pctg']==0 and value['training_rows_removed']
    print('PASS: site tables/Excel, split metadata, municipal outputs, real-station errors, MC summaries and asset hashes.')


if __name__=='__main__':
    main()
