import pandas as pd
from pathlib import Path

manifest_path = Path("data/vn/manifest.csv")
manifest = pd.read_csv(manifest_path)
print("Manifest length:", len(manifest))
print("Columns:", list(manifest.columns))

# Check Camera and B.8a
for label in ["Camera", "B.8a"]:
    sub = manifest[manifest["label_fine"] == label]
    print(f"\nLabel: {label}")
    print("Total bbox count:", len(sub))
    for sp in ["train", "val", "test"]:
        s_sp = sub[sub["split"] == sp]
        s_small = s_sp[(s_sp["x2"] - s_sp["x1"] < 8) | (s_sp["y2"] - s_sp["y1"] < 8)]
        print(f"  {sp}: total={len(s_sp)}, <8px={len(s_small)}, valid={len(s_sp) - len(s_small)}, unique_images={s_sp['img_stem'].nunique()}")

# Let's inspect all classes with prefix or unrecognized
print("\n--- Value counts of label_fine for Camera and B.8a ---")
print(manifest[manifest["label_fine"].str.contains("Camera|B.8a", case=False, na=False)]["label_fine"].value_counts())

# What about the previous audit? Why did the user say:
# "Bảng 2.2 ghi Camera 74 train, B.8a 82 train, lệch với audit trước (Camera 6 train, B.8a 150 train)."
# Let's check reports/label_audit or transcript
audit_dir = Path("reports/label_audit")
if audit_dir.exists():
    print("\nFiles in reports/label_audit:")
    for f in audit_dir.iterdir():
        print(f.name)
