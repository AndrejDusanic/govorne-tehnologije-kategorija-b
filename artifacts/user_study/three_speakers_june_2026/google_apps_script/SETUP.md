# Google Apps Script setup

Ova verzija skripte vise ne trazi rucni unos 15 pojedinacnih `driveId` vrednosti.
Dovoljno je da svi MP4 fajlovi budu u jednom Google Drive folderu i da imena budu ista kao u `video_upload_manifest.csv`.

1. Napravi Google Sheet koji ce cuvati rezultate.
2. Kopiraj ID tog Sheet-a i upisi ga u `SHEET_ID` u `Code.gs`.
3. Uploaduj svih 15 MP4 fajlova u jedan Google Drive folder.
4. Proveri da su nazivi fajlova tacno:
   `Marko1.mp4`, `Nikola3.mp4`, `Cope2.mp4`, `Nikola1.mp4`, `Marko4.mp4`, `Cope5.mp4`, `Nikola5.mp4`, `Cope1.mp4`, `Marko2.mp4`, `Nikola2.mp4`, `Cope4.mp4`, `Marko5.mp4`, `Nikola4.mp4`, `Cope3.mp4`, `Marko3.mp4`
5. Kopiraj ID Drive foldera i upisi ga u `DRIVE_FOLDER_ID` u `Code.gs`.
6. U tom folderu ukljuci `Anyone with the link -> Viewer`.
7. U Google Apps Script projektu ubaci sadrzaj `Code.gs`.
8. Deployuj kao `Web app`.
9. Stavi `Execute as: Me`.
10. Stavi `Who has access: Anyone with the link`.
11. Otvori dobijeni link i probaj anketu na telefonu.

Napomena:
- Ako neki naziv fajla ne odgovara ocekivanom imenu, skripta ce javiti gresku pri otvaranju ankete.
- Redosled prikaza videa je vec definisan u `VIDEO_ITEMS`.
