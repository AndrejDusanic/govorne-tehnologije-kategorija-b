# Izvještaj za Kategoriju B: predikcija ARKit-52 blendshape koeficijenata

## 1. Opis zadatka

Zadatak u Kategoriji B je razvoj modela koji na osnovu ulaznih informacija, odnosno audija i/ili teksta, generiše vremensku sekvencu blendshape koeficijenata za animaciju avatara. Avatar je rigovan prema ARKit-52 standardu, pa svaki izlazni frejm mora sadržati 52 vrijednosti u istom redoslijedu kao u dostavljenim `.csv` fajlovima za obuku. Izlaz se predaje kao jedan `.csv` fajl po test audio fajlu, bez headera, uz `meta.json` koji prijavljuje vrijeme inferencije po fajlu, RTF, izlazni FPS i algoritamski lookahead.

U završnoj verziji pipeline koristi audio signal kao primarni ulaz i podržava tekstualni conditioning kada je dostupan stvarni transkript. Tokom završnog prolaza ispravljen je parser transkripata tako da iz Excel fajlova čita drugu kolonu sa rečenicama, a ne prvu kolonu sa rednim brojevima uzoraka. Nakon toga su manifesti rebuildani i istreniran je novi text-aware BiGRU model. Finalni best-quality setup koristi ensemble kauzalnog audio baseline-a i novog text-aware BiGRU modela, zatim learned face refiner. Za test ZIP koji je dostupan u ovom folderu postoje samo audio fajlovi, pa je finalni local submission generisan sa praznim tekstom; validacija sa praznim tekstom je i dalje ostala bolja od starog najboljeg setupa.

## 2. Podaci i priprema

Podaci su pripremljeni skriptom `scripts/prepare_data.py`. Ona raspakuje originalne arhive, organizuje putanje do `.wav` i `.csv` fajlova, čita fonemska poravnanja, čita stvarne transkripte iz druge kolone Excel fajlova i pravi manifeste u `data/manifests/`. Skup sadrži 358 prirodnih snimaka za dva govornika (`spk08` i `spk14`) i 353 sintetizovana audio fajla. Za validaciju je korišten fiksni split sa seed vrijednošću 1337: 304 prirodna snimka za trening i 54 za validaciju. Izlazni FPS je 60, broj blendshape koeficijenata je 52, a fonemski vokabular ima 34 oznake uključujući posebne simbole.

Analiza baze je rađena skriptom `scripts/analyze_data.py`, koja generiše osnovne grafike: pregled trajanja i broja snimaka, aktivnost blendshape koeficijenata i distribuciju fonema. Ti grafici su sačuvani u `reports/figures/` i korisni su za objašnjenje zašto su usne i vilica posebno važne u funkciji gubitka. Prirodni snimci imaju ukupno oko 30.2 minuta audija, a sintetizovani oko 18.0 minuta. Budući da test faza može sadržati TTS audio, sintetizovani snimci su važni za demonstraciju i potencijalni budući domain-adaptation eksperiment.

Akustičke karakteristike se računaju klasom `AudioFeatureExtractor`. Za svaki audio signal računa se log-mel spektrogram sa 80 mel kanala, zatim prva i druga delta komponenta. Time se dobija 240-dimenzionalni feature vektor po frejmu. Feature sekvenca se interpolira na ciljani broj frejmova pri 60 FPS, kako bi bila poravnata sa blendshape sekvencom. Prije treninga se računaju statistike normalizacije nad trening splitom: srednja vrijednost i standardna devijacija za audio feature-e i za ciljne blendshape koeficijente.

## 3. Korišteni algoritmi i modeli

Osnovni model je `BlendshapeRegressor`, neuronska mreža u PyTorch-u koja prima audio feature sekvencu i ID govornika. Speaker embedding se konkatenira sa audio feature-ima da bi model mogao učiti razlike između govornika. Prvi model, `baseline_full_v1`, koristi kauzalni temporalni konvolucioni encoder (`causal_tcn`) sa dilatiranim gated residual blokovima. Ovaj režim je pogodan za real-time rad jer ne zahtijeva buduće frejmove, pa mu je algoritamski lookahead 0 ms.

Drugi model, `text_bgru_v1_cpu`, koristi bidirectional GRU temporalni encoder i tekstualni conditioning preko karakter-level GRU grane sa attention spajanjem na audio sekvencu. On može koristiti i lijevi i desni kontekst sekvence, pa daje kvalitetniju offline predikciju, ali nije strogo kauzalan. U finalnom best-quality setupu upravo ovaj model učestvuje u ensemble-u, pa lookahead treba računati kao trajanje cijelog fajla ili praktično kao veliki offline lookahead. U sačuvanom finalnom `meta.json` za test folder prijavljen je `lookahead_ms = 6609`.

Trening koristi kombinaciju nekoliko loss komponenti. Glavna regresiona komponenta je Smooth L1 loss nad normalizovanim blendshape koeficijentima. Dodatno se koristi temporalni loss nad razlikama susjednih frejmova, da se poboljša stabilnost pokreta kroz vrijeme. Za usne i vilicu se koriste veće težine, jer su ti koeficijenti najvažniji za sinhronizaciju govora i najvidljiviji na avataru. Postoji i peak-aware loss koji preko lokalnog max-pooling-a dodatno naglašava vršne pokrete kod koeficijenata kao što su `jawOpen`, `mouthFunnel`, `mouthPucker`, `mouthClose`, `mouthSmile*` i `mouthLowerDown*`. Model takođe ima pomoćnu fonemsku glavu i cross-entropy loss za predikciju fonema po frejmu iz automatskih fonemskih poravnanja.

Najbolji korišteni setup je weighted ensemble dva checkpointa: `artifacts/checkpoints/baseline_full_v1/best.pt` i `artifacts/checkpoints/text_bgru_v1_cpu/best.pt`. Tokom evaluacije i inferencije oba modela generišu predikcije u raw blendshape prostoru, zatim se kombinuju sa tezinama `0.6 / 0.4` u korist `baseline_full_v1`. Nakon toga se primjenjuje learned face refiner `artifacts/refiners/text_ensemble_weighted_face_refiner_v1.npz`. Refiner je lagani Ridge regresioni model koji koristi trenutne vrijednosti, delta vrijednosti i kvadrate vrijednosti kao ulazne karakteristike, pa blago koriguje cijelo lice bez potpunog mijenjanja osnovne predikcije. Najbolja validaciona jacina novog refiner-a je `0.15`.

Za demo u avatar aplikaciji postoji i opcioni random blink post-processing. On dodaje reproducibilna treptanja u `eyeBlinkLeft/Right`, uz male korekcije `eyeSquint*`, `browDown*` i `eyeWide*`. Ovo je korisno za prirodniji vizuelni demo, ali ga ne treba tretirati kao osnovnu validacionu metriku, jer mijenja izlaz na heuristički način.

## 4. Rezultati

Najbolji pojedinačni checkpoint po MAE je i dalje `baseline_full_v1`, ali novi `text_bgru_v1_cpu` je bolji od ranijeg `hybrid_bgru_v1` i daje bolji ensemble. Nakon novog face refiner-a dobijen je najbolji ukupni validation rezultat.

| Setup | Validation MAE | RMSE | Mouth MAE | JawOpen MAE | Napomena |
|---|---:|---:|---:|---:|---|
| `baseline_full_v1` | 0.018951 | 0.045144 | 0.023034 | 0.025855 | Najbolji pojedinačni checkpoint, kauzalni TCN |
| `hybrid_bgru_v1` | 0.020361 | 0.044822 | 0.024188 | 0.025973 | Stari BiGRU model, treniran prije ispravke transkripata |
| `text_bgru_v1_cpu` | 0.019487 | 0.043423 | 0.023047 | 0.024975 | Novi BiGRU model sa stvarnim tekstom |
| Stari ensemble + refiner | 0.018444 | 0.042886 | 0.022274 | 0.024384 | Prethodni najbolji setup |
| Weighted ensemble bez refiner-a | 0.018223 | 0.042933 | 0.021858 | 0.024047 | `baseline_full_v1 + text_bgru_v1_cpu`, tezine `0.6 / 0.4` |
| Weighted ensemble + tuned face refiner | 0.018132 | 0.042688 | 0.021741 | 0.023984 | Najbolji trenutni setup po MAE |

Najveće greške po blendshape koeficijentima ostaju na izrazito dinamičnim usnenim pokretima: `mouthLowerDownLeft/Right`, `mouthFunnel`, `mouthPucker`, `mouthSmileLeft/Right` i dijelom `mouthUpperUp*`. To je očekivano jer su ovi koeficijenti najviše zavisni od precizne fonetske i vremenske sinhronizacije. Refiner blago popravlja MAE, ali nije dramatičan skok; njegov veći doprinos je vizuelno jačanje gornjeg dijela lica i stabilnija full-face mimika. U zavrsnom prolazu dodatno su isprobani synth pseudo-labeling, benchmark `HuBERT/WavLM` feature-a i k-fold tooling. Pseudo-labeling je malo popravio samostalni text model, ali nije dao bolji finalni ensemble. `HuBERT/WavLM` su se pokazali znatno sporijim od log-mel feature-a na ovoj masini, pa nisu usvojeni kao novi default. K-fold skripte su pripremljene, ali puni k-fold retraining nije pokrenut jer bi znatno produzio eksperiment.

Finalni submit folder `artifacts/predictions/final_test_submission/` regenerisan je novim najboljim setupom i sadrži 13 `.csv` fajlova. Provjereno je da svaki `.csv` ima 52 vrijednosti po redu i da nema header. U `meta.json` su upisani `fps_out = 60`, `lookahead_ms = 6609`, `inference_time_sec`, `rtf` i tezine ensemble-a. Skripta za inferenciju sada radi jedan warm-up prije mjerenja i mjeri i pripremu ulaza po fajlu. Prosječni RTF u lokalnom CPU mjerenju za official folder je oko `0.03394`; avatar-ready verzija sa random blink postprocessingom je u `artifacts/predictions/final_test_avatar_ready/`.

## 5. Uputstvo za pokretanje

Instalacija zavisnosti:

```powershell
python -m pip install -r requirements.txt
```

Priprema podataka i analiza:

```powershell
python scripts/prepare_data.py
python scripts/analyze_data.py
```

Evaluacija najboljeg postojećeg setupa:

```powershell
python scripts/evaluate.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/text_bgru_v1_cpu/best.pt --ensemble-weights 0.6,0.4 --face-refiner artifacts/refiners/text_ensemble_weighted_face_refiner_v1.npz --device cuda --output-dir reports/figures/text_ensemble_weighted_refined
```

Trening novog offline BiGRU modela:

```powershell
python scripts/train.py --run-name improved_full_run --epochs 18 --batch-size 8 --device cuda --temporal-encoder bgru
```

Ako je potreban strogo kauzalan real-time režim, koristiti kauzalni encoder i po potrebi isključiti tekstualni kanal:

```powershell
python scripts/train.py --run-name causal_run --epochs 18 --batch-size 8 --device cuda --temporal-encoder causal_tcn --no-text-conditioning
```

Inferencija nad folderom sa novim audio fajlovima:

```powershell
python scripts/infer_folder.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/text_bgru_v1_cpu/best.pt --ensemble-weights 0.6,0.4 --face-refiner artifacts/refiners/text_ensemble_weighted_face_refiner_v1.npz --input-dir path\do\wav_foldera --output-dir artifacts/predictions/test_run --device cuda
```

Za avatar demo se može dodati treptanje:

```powershell
python scripts/infer_folder.py --checkpoint artifacts/checkpoints/baseline_full_v1/best.pt artifacts/checkpoints/text_bgru_v1_cpu/best.pt --ensemble-weights 0.6,0.4 --face-refiner artifacts/refiners/text_ensemble_weighted_face_refiner_v1.npz --input-dir path\do\wav_foldera --output-dir artifacts/predictions/test_run --device cuda --random-blinks --blink-strength 1.0
```

U Colab okruženju se može koristiti notebook `notebooks/competition_pipeline_colab.ipynb` ili bootstrap skripta:

```bash
bash scripts/colab_bootstrap.sh --with-analysis
```

## 6. Hardver, vrijeme obuke i ograničenja

Za trening se preporučuje NVIDIA GPU u Google Colab okruženju ili lokalna CUDA mašina. `baseline_full_v1` je treniran 18 epoha i u checkpoint konfiguraciji je zabilježen `device = cuda`. Novi `text_bgru_v1_cpu` je u završnom prolazu istreniran 18 epoha na ovoj CPU mašini sa 12 logičkih jezgara, jer CUDA nije bila dostupna; run je trajao približno 33 minute. Iako je CPU bio dovoljan za ovaj skup, za ponovljive eksperimente i brže iteracije i dalje je preporučen Colab GPU.

Tačno vrijeme obuke za starije checkpoint-e nije logovano, pa ga ne treba prikazivati kao izmjerenu vrijednost. Za novi text-aware BiGRU postoji lokalna CPU mjera od oko 33 minute. Razumna procjena za ponovno treniranje na Colab GPU-u je red veličine nekoliko do nekoliko desetina minuta za postojeći skup i navedene konfiguracije, uz zavisnost od dostupnog GPU-a, opterećenja Colab runtime-a i broja epoha. Zavrsni weighted ensemble i tuned refiner ne mijenjaju znacajno inference trosak u odnosu na raniji equal-weight setup, jer koriste iste checkpoint-e i isti tip refiner-a.

Važno je razlikovati dva režima rada. Best-quality offline režim koristi ensemble sa BiGRU modelom i face refiner-om; on daje najbolju validacionu metriku, ali koristi budući kontekst. Strict real-time režim treba bazirati na kauzalnom TCN modelu, jer taj model može raditi sa `lookahead_ms = 0`, uz nešto slabiji kvalitet.

## 7. Preporuke za dalji rad

Najvažniji prethodno uočeni problem, čitanje transkripata iz pogrešne Excel kolone, sada je ispravljen. Sljedeći konkretan korak je obezbijediti stvarne tekstualne transkripte i za test fajlove kad god su dostupni, jer sada model može koristiti tekst na smislen način. Za test ZIP prisutan u ovom folderu transkripti nisu bili dostupni, pa je local submission generisan sa praznim tekstom.

Drugi korak je domain adaptation na sintetizovanom govoru. U ovom prolazu je vec isproban pseudo-labeling nad `audio_synth` i kratki fine-tuning text modela, ali ta varijanta nije nadmasila finalni weighted ensemble. I dalje je smisleno nastaviti teacher-student ili DTW pristup ako bude vremena za duze treninge. Treći mogući pravac je korištenje jačeg speech backbone-a, npr. HuBERT ili WavLM, ali to povećava memorijske i vremenske zahtjeve; brzi benchmark u ovom repou je pokazao da su ovi backbone-i znatno sporiji od log-mel feature-a. Za stabilniju procjenu prije eventualnog finalnog retraining-a korisno je uvesti puni k-fold trening i posebno mjeriti rezultate po govorniku i po grupama blendshape koeficijenata.

Za mjerljiviji RTF treba dodati warm-up prije mjerenja prvog fajla ili posebno prijaviti prvi fajl kao cold-start. Za produkcijski real-time demo preporučuje se održavati odvojenu kauzalnu konfiguraciju, dok se ensemble + BiGRU koristi kao offline best-quality submission kada je primarni kriterijum prirodnost animacije.
