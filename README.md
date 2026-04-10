# Kategorija B: ARKit-52 Blendshape Prediction

Trenutno je implementiran i provjeren kompletan audio/text pipeline:

- raspakivanje i organizacija podataka
- gradnja manifesta i fiksnog `train/val` splita
- EDA grafikoni nad bazom
- hibridni audio+text model sa speaker embedding-om
- bidirectional temporal encoder za kvalitetniji offline mod
- multitask trening sa pomoćnim fonemskim loss-om
- activity/peak weighted loss za teže usne blendshape-ove
- learned face refiner koji pojacava podaktivne brow/eye/nose koeficijente bez rušenja stabilnosti
- evaluacija sa grafikonima
- inferenca koja generise `CSV` i `meta.json`
- Colab notebook za pokretanje svega iz Git repozitorija
- izvještaj za predaju u `reports/izvjestaj_kategorija_b.md` i `reports/izvjestaj_kategorija_b.docx`

## Trenutni najbolji rezultat

Najbolji setup u repou je weighted ensemble ova dva checkpointa plus tuned learned face refiner:

- `artifacts/checkpoints/baseline_full_v1/best.pt`
- `artifacts/checkpoints/text_bgru_v1_cpu/best.pt`
- tezine ensemble-a: `0.6 / 0.4`
- `artifacts/refiners/text_ensemble_weighted_face_refiner_v1.npz`

Ensemble rezultat na validation splitu:

- Validation MAE: `0.0181320`
- Validation RMSE: `0.0426877`
- Mouth-only MAE: `0.0217408`
- JawOpen MAE: `0.0239838`

Najbolji pojedinačni checkpoint po MAE je `baseline_full_v1`, dok `text_bgru_v1_cpu` služi kao drugi član ensemble-a.
Finalni local submission je `artifacts/predictions/final_test_submission/` sa `baseline_full_v1 + text_bgru_v1_cpu + weighted averaging + text_ensemble_weighted_face_refiner_v1`.


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
- `scripts/search_ensemble_weights.py`
  - pretraga najboljih težina za ensemble od dva checkpointa
- `scripts/pseudo_label_synth.py`
  - pseudo-labeling sintetizovanog audija i gradnja mješovitog manifesta
- `scripts/build_kfold_splits.py`
  - generisanje speaker-balanced k-fold splitova nad prirodnim snimcima
- `scripts/benchmark_backbones.py`
  - brzo poređenje mel vs `HuBERT` vs `WavLM` feature extraction troška
- `scripts/postprocess_blinks.py`
  - dodaje reproducibilna nasumična treptanja nad već generisanim `CSV` fajlovima za avatar demo
- `src/blendshape_project/face_refiner.py`
  - helper kod za learned full-face refinement
- `src/blendshape_project/blink_postprocess.py`
  - helper kod za random blink post-processing
- `notebooks/competition_pipeline_colab.ipynb`
  - Colab workflow
- `artifacts/checkpoints/baseline_full_v1/`
  - najbolji pojedinačni checkpoint
- `artifacts/checkpoints/hybrid_bgru_v1/`
  - prethodni hibridni `bgru` checkpoint
- `artifacts/checkpoints/text_bgru_v1_cpu/`
  - novi text-aware `bgru` checkpoint za najbolji ensemble

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
python scripts/evaluate.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/text_bgru_v1_cpu/best.pt --ensemble-weights 0.6,0.4 --face-refiner artifacts/refiners/text_ensemble_weighted_face_refiner_v1.npz --device cuda --output-dir reports/figures/text_ensemble_weighted_refined
```

4. Jaci offline modela:

```powershell
python scripts/train.py --run-name improved_full_run --epochs 18 --batch-size 8 --device cuda --temporal-encoder bgru
```

Bez novog treninga evaluacija vec prilozeni checkpoint:

```powershell
python scripts/evaluate.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt --device cuda
```

5. Evaluacija svog novog runa:

```powershell
python scripts/evaluate.py --checkpoint artifacts/checkpoints/improved_full_run/best.pt --device cuda
```

6. Inferenca nad novim audio fajlovima sa najboljim spremnim ensemble-om:

```powershell
python scripts/infer_folder.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/text_bgru_v1_cpu/best.pt --ensemble-weights 0.6,0.4 --face-refiner artifacts/refiners/text_ensemble_weighted_face_refiner_v1.npz --input-dir path\do\wav_foldera --output-dir artifacts/predictions/test_run --device cuda
```

7. Inferenca nad novim audio fajlovima sa svojim novim runom:

```powershell
python scripts/infer_folder.py --checkpoint artifacts/checkpoints/improved_full_run/best.pt --input-dir path\do\wav_foldera --output-dir artifacts/predictions/test_run --device cuda
```

8. Avatar povremeno trepce, blink post-processing direktno u inferenci:

```powershell
python scripts/infer_folder.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/text_bgru_v1_cpu/best.pt --ensemble-weights 0.6,0.4 --face-refiner artifacts/refiners/text_ensemble_weighted_face_refiner_v1.npz --input-dir path\do\wav_foldera --output-dir artifacts/predictions/test_run --device cuda --random-blinks --blink-strength 1.0
```

9. `CSV` fajlovi za blink verziju za avatar:

```powershell
python scripts/postprocess_blinks.py --input-dir artifacts/predictions/avatar_demo_synth_refined --output-dir artifacts/predictions/avatar_demo_synth_refined_blinks --audio-dir artifacts/predictions/avatar_demo_synth_refined
```




GitHub i Colab:

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
- `artifacts/checkpoints/text_bgru_v1_cpu/`
- `artifacts/refiners/`
- `reports/izvjestaj_kategorija_b.md`
- `reports/izvjestaj_kategorija_b.docx`

U repou ne treba da budu:

- `data/extracted/`
- `artifacts/tmp/`
- lokalni avatar demo outputi
- cache folderi



```powershell
python scripts/setup_github_remote.py --username YOUR_GITHUB_USERNAME
```


```powershell
python scripts/setup_github_remote.py --username YOUR_GITHUB_USERNAME --push
```


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


Rucno:

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
!python scripts/evaluate.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/text_bgru_v1_cpu/best.pt --ensemble-weights 0.6,0.4 --face-refiner artifacts/refiners/text_ensemble_weighted_face_refiner_v1.npz --device cuda --output-dir reports/figures/text_ensemble_weighted_refined
```

Ako hoces trening u Colabu:

```bash
!python scripts/train.py --run-name improved_full_run --epochs 18 --batch-size 8 --device cuda --temporal-encoder bgru
```

Ako hoces inferencu nad test WAV folderom:

```bash
!python scripts/infer_folder.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/text_bgru_v1_cpu/best.pt --ensemble-weights 0.6,0.4 --face-refiner artifacts/refiners/text_ensemble_weighted_face_refiner_v1.npz --input-dir test_wavs --output-dir artifacts/predictions/colab_test --device cuda
```

## Model u jednoj recenici

Novi default setup koristi log-mel + delta + delta-delta audio feature-e na `60 FPS`, speaker conditioning, stvarni text conditioning preko karakter-level attention grane u `text_bgru_v1_cpu`, bidirectional GRU temporal encoder, dodatni fonemski supervision, peak-aware loss za usne koeficijente, weighted averaging `0.6 / 0.4` i learned face refiner za jacu mimiku cijelog lica.

`scripts/evaluate.py` i `scripts/infer_folder.py` sada podrzavaju vise `--checkpoint` argumenata i rade raw-space averaging, a zatim opciono i face refinement, pa dobijes ensemble bez dodatnog koda.

##  Unapredjenja

- pseudo-labeling nad `audio_synth`
- jaci multimodal encoder tipa Conformer ili pretrained speech backbone (`HuBERT`/`WavLM`)
- speaker- i phoneme-aware refiner treniran i na sintetickom domenu
- dva moda: `offline best-quality` i `strict causal low-latency`
- k-fold validacija prije finalnog treninga
- dodavanje stvarnih `.txt` transkripata za test fajlove kada su dostupni, jer novi model sada stvarno koristi tekst

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
python scripts/infer_folder.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/text_bgru_v1_cpu/best.pt --ensemble-weights 0.6,0.4 --face-refiner artifacts/refiners/text_ensemble_weighted_face_refiner_v1.npz --input-dir test_wavs --text-dir test_texts --output-dir artifacts/predictions/test_run --device cuda
```

## Eksperimenti iz zavrsnog prolaza

- `weighted ensemble + tuned refiner` je donio novi najbolji validation `MAE` bez primjetnog rasta inference troska; finalni local submission sada ima prosjecni CPU `RTF ~ 0.03394`
- `synth/domain adaptation` je isproban preko pseudo-labeling-a nad `audio_synth` i kratkog fine-tuning-a (`text_bgru_v1_synth_ft`), ali nije nadmasio finalni ensemble
- `HuBERT` i `WavLM` su benchmarkovani skriptom `scripts/benchmark_backbones.py`; na ovoj masini su feature-i bili mnogo sporiji od mel feature-a, pa nisu usvojeni kao novi default
- `k-fold` tooling je dodat, ali puni k-fold retraining nije pokrenut jer bi znacajno produzio eksperiment bez garancije da ce biti bolji od finalnog submit-ready setupa
- pravi test transkripti i dalje nisu dostupni u `test_set_catB.zip`, pa inferenca nad tim folderom radi sa praznim tekstom; kad ti transkripti postoje, novi text-aware model ih moze direktno koristiti
