# Three-Speaker User Study Bundle

Ovaj bundle je pripremljen za mobilno ocenjivanje kvaliteta avatar animacije nad tri uslova:

- `Nikola`: vidjeni govornik (`seen`)
- `Marko`: nevidjeni govornik (`unseen`)
- `Cope`: TTS (`tts`)

## Sta je unutra

- `audio_inputs/`: 15 WAV fajlova preimenovanih za studiju
- `text_inputs/`: 15 TXT fajlova sa tekstom za inference
- `predictions/`: generisani CSV + meta nakon inference
- `avatar_ready/`: upareni WAV + CSV fajlovi za FTNFacialRig
- `google_apps_script/Code.gs`: mobilni survey template za Google Apps Script
- `video_upload_manifest.csv`: spisak videa i mesta gde treba upisati Google Drive ID

## Vazna napomena

Tekst za `Nikola` set je povucen tacno iz `spk03` transkripta jer su FC fajlovi identifikovani kao originalni snimci tog govornika.
Tekst za `Marko` i `Cope` set je trenutno postavljen pod pretpostavkom da pratite istih 5 recenica u istom redosledu kao `Nikola`.
Ako to nije tacno, samo izmeni odgovarajuce `.txt` fajlove u `text_inputs/` i ponovo pokreni inference.

## Kako dalje

1. Otvori FTNFacialRig i za svaki fajl iz `avatar_ready/` snimi MP4 ekran.
2. Uploaduj 15 MP4 fajlova na Google Drive.
3. Upisi njihove `drive_file_id` vrednosti u `video_upload_manifest.csv` ili direktno u `google_apps_script/Code.gs`.
4. Deployuj Apps Script kao web app i posalji link ispitanicima.

Ako promenis neke `.txt` fajlove, ponovo pokreni inference ovom komandom:

```powershell
python scripts/infer_folder.py --checkpoint artifacts/checkpoints/expanded_bgru_no_speaker_v1/best.pt --face-refiner artifacts/refiners/expanded_bgru_no_speaker_v1_face_refiner.npz --input-dir artifacts/user_study/three_speakers_june_2026/audio_inputs --text-dir artifacts/user_study/three_speakers_june_2026/text_inputs --output-dir artifacts/user_study/three_speakers_june_2026/predictions --device cpu --default-speaker spk03 --random-blinks --blink-strength 1.0
```

## Stavke u studiji

- `Nikola1` -> `nikola_01.wav/.csv` (seen)
- `Nikola2` -> `nikola_02.wav/.csv` (seen)
- `Nikola3` -> `nikola_03.wav/.csv` (seen)
- `Nikola4` -> `nikola_04.wav/.csv` (seen)
- `Nikola5` -> `nikola_05.wav/.csv` (seen)
- `Marko1` -> `marko_01.wav/.csv` (unseen)
- `Marko2` -> `marko_02.wav/.csv` (unseen)
- `Marko3` -> `marko_03.wav/.csv` (unseen)
- `Marko4` -> `marko_04.wav/.csv` (unseen)
- `Marko5` -> `marko_05.wav/.csv` (unseen)
- `Cope1` -> `cope_01.wav/.csv` (tts)
- `Cope2` -> `cope_02.wav/.csv` (tts)
- `Cope3` -> `cope_03.wav/.csv` (tts)
- `Cope4` -> `cope_04.wav/.csv` (tts)
- `Cope5` -> `cope_05.wav/.csv` (tts)
