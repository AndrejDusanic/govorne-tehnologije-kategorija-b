# Kategorija B: ARKit-52 Blendshape Prediction

Ovaj folder je pretvoren u kompletan takmicarski projekat za zadatak iz `kategorijaB_baza (2).pdf`: iz teksta i/ili audija predvidjeti ARKit-52 blendshape sekvencu pogodnu za avatar i real-time rad.

Trenutno je implementiran i provjeren kompletan audio-driven pipeline:

- raspakivanje i organizacija podataka
- gradnja manifesta i fiksnog `train/val` splita
- EDA grafikoni nad bazom
- causal audio model sa speaker embedding-om
- multitask trening sa pomocnim fonemskim loss-om
- evaluacija sa grafikonima
- inferenca koja generise `CSV` i `meta.json`
- Colab notebook za pokretanje svega iz Git repozitorija

## Trenutni najbolji rezultat

Run: `artifacts/checkpoints/baseline_full_v1`

- Validation MAE: `0.0189505`
- Validation RMSE: `0.0451437`
- Mouth-only MAE: `0.0230344`
- JawOpen MAE: `0.0258554`

Najvazniji grafici su vec generisani:

- `reports/figures/dataset_overview.png`
- `reports/figures/blendshape_activity.png`
- `reports/figures/phoneme_distribution.png`
- `artifacts/checkpoints/baseline_full_v1/training_curves.png`
- `artifacts/checkpoints/baseline_full_v1/val_per_blendshape_mae.png`
- `artifacts/checkpoints/baseline_full_v1/spk08_001_overlay.png`

## Struktura

- `src/blendshape_project/`
  - kod za feature extraction, dataset, model i evaluaciju
- `scripts/prepare_data.py`
  - raspakuje arhive i pravi manifeste
- `scripts/analyze_data.py`
  - pravi EDA grafike
- `scripts/train.py`
  - trenira model i cuva checkpoint + grafike
- `scripts/evaluate.py`
  - evaluacija nad validation splitom
- `scripts/infer_folder.py`
  - inferenca nad folderom sa `.wav` fajlovima
- `notebooks/competition_pipeline_colab.ipynb`
  - Colab workflow
- `artifacts/checkpoints/baseline_full_v1/`
  - najbolji istrenirani model i metrike

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

3. Trening:

```powershell
python scripts/train.py --run-name baseline_full_v1 --epochs 18 --batch-size 8 --device cuda
```

4. Evaluacija:

```powershell
python scripts/evaluate.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt --device cuda
```

5. Inferenca nad novim audio fajlovima:

```powershell
python scripts/infer_folder.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt --input-dir path\do\wav_foldera --output-dir artifacts/predictions/test_run --device cuda
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

## Model u jednoj recenici

Model koristi log-mel + delta + delta-delta audio feature-e na `60 FPS`, speaker conditioning i causal dilated temporal blokove, uz dodatni fonemski supervision iz `labels_aligned` skupa.

## Ideje za jos bolji plasman

- pseudo-labeling nad `audio_synth`
- dodatni smoothing/post-processing po blendshape grupama
- poseban loss za lipsync koeficijente sa vecim tezinama
- dva moda: `offline best-quality` i `strict causal low-latency`
- k-fold validacija prije finalnog treninga

