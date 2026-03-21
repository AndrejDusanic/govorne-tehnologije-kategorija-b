# Kategorija B: ARKit-52 Blendshape Prediction

Ovaj folder je pretvoren u kompletan takmicarski projekat za zadatak iz `kategorijaB_baza (2).pdf`: iz teksta i/ili audija predvidjeti ARKit-52 blendshape sekvencu pogodnu za avatar i real-time rad.

Trenutno je implementiran i provjeren kompletan audio-driven pipeline:

- raspakivanje i organizacija podataka
- gradnja manifesta i fiksnog `train/val` splita
- EDA grafikoni nad bazom
- hibridni audio+text model sa speaker embedding-om
- bidirectional temporal encoder za kvalitetniji offline mod
- multitask trening sa pomocnim fonemskim loss-om
- activity/peak weighted loss za teze usne blendshape-ove
- learned face refiner koji pojacava podaktivne brow/eye/nose koeficijente bez rusenja stabilnosti
- evaluacija sa grafikonima
- inferenca koja generise `CSV` i `meta.json`
- Colab notebook za pokretanje svega iz Git repozitorija

## Trenutni najbolji rezultat

Najbolji setup koji je trenutno spreman u repou je ensemble ova dva checkpointa plus learned face refiner:

- `artifacts/checkpoints/baseline_full_v1/best.pt`
- `artifacts/checkpoints/hybrid_bgru_v1/best.pt`
- `artifacts/refiners/ensemble_face_refiner_v1.npz`

Ensemble rezultat na validation splitu:

- Validation MAE: `0.0184445`
- Validation RMSE: `0.0428856`
- Mouth-only MAE: `0.0222742`
- JawOpen MAE: `0.0243837`

Najbolji pojedinacni checkpoint po MAE i dalje je `baseline_full_v1`, dok `hybrid_bgru_v1` sluzi kao drugi clan ensemble-a i popravlja ukupan skor kada se prosjece raw predikcije. Face refiner se zatim primjenjuje nad ensemble izlazom i dize dinamiku gornjeg dijela lica, posebno za `browDown*`, `eyeBlink*`, `eyeLookDown*` i `noseSneer*`.

Najvazniji grafici su vec generisani:

- `reports/figures/dataset_overview.png`
- `reports/figures/blendshape_activity.png`
- `reports/figures/phoneme_distribution.png`
- `reports/figures/ensemble_refined_default/validation_per_blendshape_mae.png`
- `reports/figures/ensemble_refined_default/ensemble_spk08_001_overlay.png`
- `reports/figures/ensemble_default/validation_per_blendshape_mae.png`
- `reports/figures/ensemble_default/ensemble_spk08_001_overlay.png`
- `artifacts/checkpoints/baseline_full_v1/training_curves.png`
- `artifacts/checkpoints/baseline_full_v1/val_per_blendshape_mae.png`
- `artifacts/checkpoints/baseline_full_v1/spk08_001_overlay.png`
- `artifacts/checkpoints/hybrid_bgru_v1/training_curves.png`
- `artifacts/checkpoints/hybrid_bgru_v1/val_per_blendshape_mae.png`

## Struktura

- `src/blendshape_project/`
  - kod za feature extraction, dataset, model i evaluaciju
- `scripts/prepare_data.py`
  - raspakuje arhive i pravi manifeste
- `scripts/analyze_data.py`
  - pravi EDA grafike
- `scripts/train.py`
  - trenira model i cuva checkpoint + grafike
- `scripts/train_face_refiner.py`
  - trenira learned post-processing refiner nad jednim ili vise checkpointa
- `scripts/evaluate.py`
  - evaluacija nad validation splitom za jedan checkpoint ili ensemble, uz opcioni face refiner
- `scripts/infer_folder.py`
  - inferenca nad folderom sa `.wav` fajlovima za jedan checkpoint ili ensemble, uz opcioni face refiner
- `src/blendshape_project/face_refiner.py`
  - helper kod za learned full-face refinement
- `notebooks/competition_pipeline_colab.ipynb`
  - Colab workflow
- `artifacts/checkpoints/baseline_full_v1/`
  - najbolji pojedinacni checkpoint
- `artifacts/checkpoints/hybrid_bgru_v1/`
  - hibridni `bgru` checkpoint za ensemble
- `artifacts/refiners/`
  - spremljen face refiner i njegove metrike

## Brzi start lokalno

1. Instalacija:

```powershell
python -m pip install -r requirements.txt
```

2. Priprema podataka:

```powershell
python scripts/prepare_data.py
python scripts/analyze_data.py
```

3. Najbrza provjera najboljeg spremnog setupa:

```powershell
python scripts/evaluate.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/hybrid_bgru_v1/best.pt --face-refiner artifacts/refiners/ensemble_face_refiner_v1.npz --device cuda
```

4. Trening novog jaceg offline modela:

```powershell
python scripts/train.py --run-name improved_full_run --epochs 18 --batch-size 8 --device cuda --temporal-encoder bgru
```

Ako hoces samo da provjeris da setup radi bez novog treninga, evaluiraj vec prilozeni checkpoint:

```powershell
python scripts/evaluate.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt --device cuda
```

5. Evaluacija svog novog runa:

```powershell
python scripts/evaluate.py --checkpoint artifacts/checkpoints/improved_full_run/best.pt --device cuda
```

6. Inferenca nad novim audio fajlovima sa najboljim spremnim ensemble-om:

```powershell
python scripts/infer_folder.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/hybrid_bgru_v1/best.pt --face-refiner artifacts/refiners/ensemble_face_refiner_v1.npz --input-dir path\do\wav_foldera --output-dir artifacts/predictions/test_run --device cuda
```

7. Inferenca nad novim audio fajlovima sa svojim novim runom:

```powershell
python scripts/infer_folder.py --checkpoint artifacts/checkpoints/improved_full_run/best.pt --input-dir path\do\wav_foldera --output-dir artifacts/predictions/test_run --device cuda
```

## Colab workflow

U Colabu koristi `notebooks/competition_pipeline_colab.ipynb`.

Bitna napomena za Git:

- u ovom folderu postoje veliki fajlovi
- zato je repo pripremljen za `git-lfs`
- originalne arhive ostaju source-of-truth
- `data/extracted/` se ne verzionise da ne duplira ogromne podatke

Ako budes radio iz GitHub-a i Colaba:

```bash
git lfs install
git lfs pull
```

## GitHub private repo setup

Predlozeni naziv private repoa:

- `govorne-tehnologije-kategorija-b`

U repou treba da ostanu:

- `src/`
- `scripts/`
- `notebooks/`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `.gitattributes`
- originalni ZIP fajlovi i PDF
- `artifacts/checkpoints/baseline_full_v1/`
- `artifacts/checkpoints/hybrid_bgru_v1/`
- `artifacts/refiners/`

U repou ne treba da budu:

- `data/extracted/`
- `artifacts/tmp/`
- lokalni avatar demo outputi
- cache folderi

Za lokalno povezivanje na GitHub koristi helper skriptu:

```powershell
python scripts/setup_github_remote.py --username YOUR_GITHUB_USERNAME
```

Ako hoces odmah i push:

```powershell
python scripts/setup_github_remote.py --username YOUR_GITHUB_USERNAME --push
```

Ako zelis samo da vidis koji ce URL biti postavljen:

```powershell
python scripts/setup_github_remote.py --username YOUR_GITHUB_USERNAME --print-only
```

## Colab quick start

Najbrzi nacin nakon kloniranja repoa u Colabu:

```bash
bash scripts/colab_bootstrap.sh --with-analysis
```

To radi:

- `git lfs install`
- `git lfs pull`
- `pip install -r requirements.txt`
- `python scripts/prepare_data.py`
- opciono `python scripts/analyze_data.py`

Ako hoces rucno:

```bash
!apt-get -qq update
!apt-get -qq install -y git-lfs
!git lfs install
!git lfs pull
!python -m pip install -r requirements.txt
!python scripts/prepare_data.py
!python scripts/analyze_data.py
```

Ako hoces samo evaluaciju vec istreniranog modela:

```bash
!python scripts/evaluate.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/hybrid_bgru_v1/best.pt --face-refiner artifacts/refiners/ensemble_face_refiner_v1.npz --device cuda
```

Ako hoces trening u Colabu:

```bash
!python scripts/train.py --run-name improved_full_run --epochs 18 --batch-size 8 --device cuda --temporal-encoder bgru
```

Ako hoces inferencu nad test WAV folderom:

```bash
!python scripts/infer_folder.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/hybrid_bgru_v1/best.pt --face-refiner artifacts/refiners/ensemble_face_refiner_v1.npz --input-dir test_wavs --output-dir artifacts/predictions/colab_test --device cuda
```

## Model u jednoj recenici

Novi default model koristi log-mel + delta + delta-delta audio feature-e na `60 FPS`, speaker conditioning, text conditioning preko karakter-level attention grane i bidirectional GRU temporal encoder, uz dodatni fonemski supervision, peak-aware loss za usne koeficijente i learned face refiner za jacu mimiku cijelog lica.

`scripts/evaluate.py` i `scripts/infer_folder.py` sada podrzavaju vise `--checkpoint` argumenata i rade raw-space averaging, a zatim opciono i face refinement, pa dobijes ensemble bez dodatnog koda.

## Ideje za jos bolji plasman

- pseudo-labeling nad `audio_synth`
- jaci multimodal encoder tipa Conformer ili pretrained speech backbone (`HuBERT`/`WavLM`)
- speaker- i phoneme-aware refiner treniran i na sintetickom domenu
- dva moda: `offline best-quality` i `strict causal low-latency`
- k-fold validacija prije finalnog treninga

## Novi trening komande

Za novi kvalitetniji offline model:

```powershell
python scripts/train.py --run-name improved_full_run --epochs 18 --batch-size 8 --device cuda --temporal-encoder bgru
```

Ako zelis stari strogo kauzalni mod:

```powershell
python scripts/train.py --run-name causal_run --epochs 18 --batch-size 8 --device cuda --temporal-encoder causal_tcn --no-text-conditioning
```

## Inferenca sa tekstom

Ako uz audio imas i tekst po fajlu, napravi folder sa `.txt` fajlovima istog naziva kao `.wav`.

Primjer:

- `test_wavs/spk08_test.wav`
- `test_texts/spk08_test.txt`

Komanda:

```powershell
python scripts/infer_folder.py --checkpoint artifacts/checkpoints/improved_full_run/best.pt --input-dir test_wavs --text-dir test_texts --output-dir artifacts/predictions/test_run --device cuda
```
