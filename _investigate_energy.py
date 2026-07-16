import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from pathlib import Path
from src import preprocessing, feature_engineering as fe, recommendation_engine as rec
from src import bed_snapshot as bs

cleaned = preprocessing.clean_all(); feats = fe.build_all(cleaned)
outputs, cards = rec.recommend(cleaned, feats)

print('=== energy_anomaly output ===')
en = outputs.get('energy_anomaly')
print('type/len:', type(en).__name__, None if en is None else len(en))
print(en.to_string(index=False) if en is not None and len(en) else '(empty)')

print('\n=== electricity_alert ===')
ea = outputs.get('electricity_alert')
print('type/len:', type(ea).__name__, None if ea is None else len(ea))
print(ea.to_string(index=False) if ea is not None and len(ea) else '(empty)')

print('\n=== Energy/Electricity cards ===')
for c in cards:
    if 'Energy' in c.get('category','') or 'Electricity' in c.get('category',''):
        print(c)

snap = bs.live_bed_snapshot(cleaned['bookings'])
for code in ['C13','C13A','C13B']:
    c = snap[snap['apartment_code']==code]
    if len(c):
        print(f'\n=== {code} live_status ===')
        print(c[['apartment_code','bed_code','live_status']].to_string(index=False))
        print('value_counts', c['live_status'].value_counts().to_dict())
        tenant_states=['Occupied','Notice','Notice-Booked']
        occ_now = int(c['live_status'].isin(tenant_states).sum())
        print('occupied_now:', occ_now)

elec = cleaned['electricity']
print('\n=== apartments containing 13 in code ===')
print(sorted(elec['apartment_code'].astype(str).unique()))
for code in sorted(set(elec['apartment_code'].astype(str))):
    if '13' in code or code.startswith('C1'):
        rows = elec[elec['apartment_code']==code].sort_values('billing_period')
        print(code, 'rows', len(rows), 'latest', rows.tail(1)[['billing_period','units_consumed','amount']].to_string(index=False) if len(rows) else 'none')

# anomalies csv
p = Path('outputs/anomalies_electricity.csv')
if p.exists():
    adf = pd.read_csv(p)
    print('\nanomalies_electricity columns', list(adf.columns), 'rows', len(adf))
    print('unique apartments sample', sorted(adf['apartment_code'].astype(str).unique())[:40])
    hit = adf[adf['apartment_code'].astype(str).str.contains('C13', na=False)]
    print('C13-like anomaly rows:', len(hit))
    print(hit.head(10).to_string(index=False) if len(hit) else 'none')
    # also check anomaly==1 for any C1x
    print('top anomaly apartments:')
    print(adf[adf.get('anomaly',1)==1].groupby('apartment_code').size().sort_values(ascending=False).head(15).to_string() if 'anomaly' in adf.columns else adf.groupby('apartment_code').size().sort_values(ascending=False).head(15).to_string())
