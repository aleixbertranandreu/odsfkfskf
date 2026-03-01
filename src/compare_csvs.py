import pandas as pd

df1 = pd.read_csv("submission_fashionclip_finetuned.csv")
df2 = pd.read_csv("GRAND_ENSEMBLE_HackUDC_submission.csv")

print(f"Original shape: {df1.shape}")
print(f"Grand Ensemble shape: {df2.shape}")

if df1.equals(df2):
    print("❌ ERROR CRÍTICO: El Grand Ensemble es BIT A BIT IDENTICO al CSV original.")
else:
    print("✅ Los CSV son diferentes. Calculando divergencia...")
    
    # Check how many top-1 predictions changed per bundle
    b1 = df1.groupby('bundle_asset_id').first().reset_index()
    b2 = df2.groupby('bundle_asset_id').first().reset_index()
    
    merged = pd.merge(b1, b2, on='bundle_asset_id', suffixes=('_orig', '_ens'))
    changed_top1 = (merged['product_asset_id_orig'] != merged['product_asset_id_ens']).sum()
    
    print(f"Top-1 changes: {changed_top1} out of {len(b1)} bundles ({(changed_top1/len(b1))*100:.2f}%)")
    
    # Check Top-15 intersection (MRR@15 metric)
    intersection_scores = []
    g1 = df1.groupby('bundle_asset_id')['product_asset_id'].apply(list)
    g2 = df2.groupby('bundle_asset_id')['product_asset_id'].apply(list)
    
    for b_id in g1.index:
        if b_id in g2:
            top15_1 = set(g1[b_id][:15])
            top15_2 = set(g2[b_id][:15])
            overlap = len(top15_1.intersection(top15_2))
            intersection_scores.append(overlap / 15.0)
            
    avg_overlap = sum(intersection_scores) / len(intersection_scores)
    print(f"Average overlap in Top-15 predictions: {avg_overlap*100:.2f}%")
