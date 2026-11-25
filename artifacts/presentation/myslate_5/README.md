# MySlate 5: avatar za prezentaciju

Izvor je `20261125_MySlate_5.zip`, odnosno iPhone MOV sa govorom. Audio je izdvojen kao mono WAV na 16 kHz. `avatar_ready/myslate_5.csv` su predvidjeni ARKit-52 koeficijenti na 60 FPS, dobijeni modelom `expanded_bgru_no_speaker_v1` uz face refiner i treptanje.

U `FTNFacialRig` izaberi `avatar_ready/myslate_5.csv` za **CSV File**, `avatar_ready/myslate_5.wav` za **Sound File**, postavi **Frame Rate** na `60` i klikni **Run**. Gornjim levim dugmetom skloni bocni panel tokom snimanja.

Prepis je u `text_inputs/myslate_5.txt`. Automatski je dobijen iz snimka i rucno su ispravljene ocigledne greske, pa proveri tekst ako ti je potrebna potpuno doslovna transkripcija.

Ako ponovo generises CSV, koristi `--text-dir artifacts/presentation/myslate_5/text_inputs`. Aplikacija tokom reprodukcije sama generise `avatar_ready/myslate_5.txt` kao log koeficijenata, pa taj fajl nije transkript.
