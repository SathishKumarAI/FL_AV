# Getting the BDD100K images

The repo ships every shard's **labels** and split lists. It does not ship the JPEGs.
This is the only page you should need to get them.

```
batch/batch_1 .. batch_10/
├── labels/{train,val}/*.txt   ← committed (6 308 train, 1 010 val per shard)
├── train.txt val.txt test.txt ← committed
├── data.yaml                  ← committed (nc: 13)
└── images/{train,val}/        ← EMPTY until you do the below
```

## Do this

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('solesensei/solesensei_bdd100k'))"
```

**No Kaggle account or API token is needed** — the dataset is public and `kagglehub`
downloads it anonymously. It prints the extracted path and caches under
`~/.cache/kagglehub/` (`%USERPROFILE%\.cache\kagglehub` on Windows). ~6.5 GB, and it
resumes if interrupted.

Then point the populator at the directory holding `train/` and `val/`:

```bash
cd my-project
python scripts/populate_images.py --pool <printed-path>/bdd100k/images/100k
```

The script **hardlinks** by default, so the images cost no extra disk — but the pool
and the repo must be on the same volume. Across volumes, pass `--copy`. It is
idempotent, and it deletes the stale Ultralytics `labels/*.cache` files that would
otherwise resolve paths from whichever machine built them.

Smaller first pass, if you just want something running:

```bash
python scripts/populate_images.py --pool <pool> --batches 1,2 --limit 200
```

Verify:

```bash
ls my-project/batch/batch_1/images/train | wc -l     # 6308 for a full populate
python -c "from my_project.task import count_shard_examples; print(count_shard_examples(1,'train'))"
```

`count_shard_examples` counts the images actually present, not the entries in
`train.txt`. If the two disagree the count follows the disk — that number is FedAvg's
aggregation weight, and Ultralytics trains on the files, not on the manifest.

## Dead ends — do not spend time here again

Every one of these was checked and ruled out on **2026-08-04**. They are listed so
nobody re-runs the search.

| Where | What happened |
|---|---|
| The README's old "Download Dataset" Drive link (folder `1R-lelZR3LBgeHfMlRR_OhOIzfUuxPBcZ`) | **Dead.** "requested entity was not found" — deleted, or owned by an account that no longer shares it. The link has been removed from the README; do not restore it. |
| Google Drive, `sathishkumar786.ml@gmail.com` | **Empty.** Every folder and every archive mime-type enumerated: zero zip/tar/7z anywhere in the account, no folder named bdd/yolo/dataset/FL_AV. Doc-type files only. |
| ETH mirror `dl.cv.ethz.ch/bdd100k/data/` (listed in the BDD100K docs) | **Did not resolve.** Confirm it is back before relying on it. |
| `origin/images` branch | **0 images**, despite the name. |
| `origin/laptop_copy` branch | 450 JPEGs — 10 per split per shard. A toy fixture, **not** the dataset. It is genuinely useful as a CI fixture (the `simulation-smoke` job uses it) and useless for training. |
| `C:\Users\sathish\Downloads\FL_ModelForAV\` | The path baked into the committed `data.yaml` and `full_data_run/detect/train2/args.yaml`. A previous machine, not reachable from here. |

## Alternatives to Kaggle

Only if the Kaggle dataset disappears:

- Official — <https://bdd-data.berkeley.edu/>, then *100K Images*. Free, but it
  **does** require an account, which is why `kagglehub` is the documented path.

Either way you only need `bdd100k/images/100k/{train,val}/`. Ignore the `labels/` in
those archives — they are raw BDD JSON, and this repo's YOLO `.txt` labels are already
converted and committed. Re-converting them would overwrite good data with worse.

The val set alone (~1 GB) is enough for a smoke run; the full 100K set (~5.3 GB train)
is needed for real training.
