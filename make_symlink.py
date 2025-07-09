import glob
import os
from tqdm import tqdm

OUTDIR = "/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/junyan/robot/droid/droid_raw/processed_0421_symlink/"

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    # files = glob.glob("/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/junyan/robot/droid/droid_raw/processed_0421/**/frames/3dfeat", recursive=True) + glob.glob("/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/junyan/robot/droid/droid_raw/processed_0421/**/actions", recursive=True)
    files = glob.glob("/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/junyan/robot/droid/droid_raw/processed_0421/**/actions", recursive=True)
    for file in tqdm(files):
        outdir = file.replace("processed_0421", "symlink/processed_0421")
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        os.symlink(file, outdir)
        tqdm.write(f"symlink {file} -> {outdir}")
    print(f"Done, {len(files)} files")
