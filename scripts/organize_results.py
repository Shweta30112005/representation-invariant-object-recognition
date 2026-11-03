import os
import shutil
from pathlib import Path

def organize_results(base_path):
    base = Path(base_path)
    csvs_dir = base / "csvs"
    tables_dir = base / "tables"
    vis_dir = base / "visualizations"
    cam_dir = base / "activation_maps"

    for d in [csvs_dir, tables_dir, vis_dir, cam_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Move raw benchmark CSVs from base into csvs/
    for f in base.glob("*.csv"):
        dest = csvs_dir / f.name
        shutil.move(str(f), str(dest))
        print(f"Moved {f.name} -> csvs/")

    # Move any .png files from tables/ into visualizations/
    for f in tables_dir.glob("*.png"):
        dest = vis_dir / f.name
        shutil.move(str(f), str(dest))
        print(f"Moved {f.name} -> visualizations/")

    # Move plots/ contents into visualizations/ if plots exists
    plots_dir = base / "plots"
    if plots_dir.exists():
        for f in plots_dir.glob("*"):
            dest = vis_dir / f.name
            shutil.move(str(f), str(dest))
        try:
            plots_dir.rmdir()
        except Exception:
            pass

    # If activation_maps_all exists, merge/move into activation_maps/
    cam_all = base / "activation_maps_all"
    if cam_all.exists() and cam_all != cam_dir:
        for item in cam_all.glob("*"):
            dest = cam_dir / item.name
            if dest.exists() and dest.is_dir():
                shutil.rmtree(str(dest))
            shutil.move(str(item), str(dest))
            print(f"Moved activation maps: {item.name} -> activation_maps/")

    print("\nSummary of results directory:")
    for sub in [csvs_dir, tables_dir, vis_dir, cam_dir]:
        count = len(list(sub.glob("*")))
        print(f"  {sub.name}/: {count} items")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results"
    organize_results(path)
