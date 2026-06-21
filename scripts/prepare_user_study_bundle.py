from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


PROMPT_NUMBERS = [29, 46, 114, 143, 147]
PROMPT_INDEX_TO_FILE_INDEX = {prompt: index + 1 for index, prompt in enumerate(PROMPT_NUMBERS)}
SURVEY_ORDER = [
    "marko_01",
    "nikola_03",
    "cope_02",
    "nikola_01",
    "marko_04",
    "cope_05",
    "nikola_05",
    "cope_01",
    "marko_02",
    "nikola_02",
    "cope_04",
    "marko_05",
    "nikola_04",
    "cope_03",
    "marko_03",
]


def load_prompt_texts() -> dict[int, str]:
    transcript_path = ROOT / "data" / "extracted" / "spk03_blendshapes" / "spk03_transcript.xlsx"
    frame = pd.read_excel(transcript_path, header=None, engine="openpyxl")
    texts: dict[int, str] = {}
    for prompt_number in PROMPT_NUMBERS:
        row = frame[frame.iloc[:, 0] == prompt_number]
        if row.empty:
            raise ValueError(f"Prompt {prompt_number} was not found in {transcript_path}")
        texts[prompt_number] = str(row.iloc[0, 1]).strip()
    return texts


def build_items(prompt_texts: dict[int, str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    nikola_dir = Path(r"C:\Users\Administrator\Downloads\Nikola")
    stefan_dir = Path(r"C:\Users\Administrator\Downloads\Stefan")
    cope_dir = Path(r"C:\Users\Administrator\Downloads\COPE")

    for prompt_number in PROMPT_NUMBERS:
        file_index = PROMPT_INDEX_TO_FILE_INDEX[prompt_number]
        items.append(
            {
                "key": f"nikola_{file_index:02d}",
                "label": f"Nikola{file_index}",
                "condition": "seen",
                "speaker_name": "Nikola",
                "source_audio": str((nikola_dir / f"FC_{prompt_number:03d}.wav").resolve()),
                "prompt_number": str(prompt_number),
                "text": prompt_texts[prompt_number],
                "text_origin": "spk03 transcript (exact match to FC file)",
            }
        )

    for file_index, prompt_number in enumerate(PROMPT_NUMBERS, start=1):
        items.append(
            {
                "key": f"marko_{file_index:02d}",
                "label": f"Marko{file_index}",
                "condition": "unseen",
                "speaker_name": "Marko",
                "source_audio": str((stefan_dir / f"stefan_{file_index - 1:04d}.wav").resolve()),
                "prompt_number": str(prompt_number),
                "text": prompt_texts[prompt_number],
                "text_origin": "assumed same prompt order as Nikola set",
            }
        )

    for file_index, prompt_number in enumerate(PROMPT_NUMBERS, start=1):
        items.append(
            {
                "key": f"cope_{file_index:02d}",
                "label": f"Cope{file_index}",
                "condition": "tts",
                "speaker_name": "Cope",
                "source_audio": str((cope_dir / f"Cope{file_index}.wav").resolve()),
                "prompt_number": str(prompt_number),
                "text": prompt_texts[prompt_number],
                "text_origin": "assumed same prompt order as Nikola set",
            }
        )

    return items


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def prepare_bundle(output_root: Path) -> list[dict[str, str]]:
    prompt_texts = load_prompt_texts()
    items = build_items(prompt_texts)

    audio_dir = ensure_dir(output_root / "audio_inputs")
    text_dir = ensure_dir(output_root / "text_inputs")

    for item in items:
        source_audio = Path(item["source_audio"])
        audio_target = audio_dir / f"{item['key']}.wav"
        shutil.copy2(source_audio, audio_target)
        item["audio_path"] = str(audio_target.resolve())

        text_target = text_dir / f"{item['key']}.txt"
        write_text(text_target, item["text"])
        item["text_path"] = str(text_target.resolve())

    manifest = {
        "study_name": "three_speakers_mobile_rating_june_2026",
        "description": "Seen vs unseen vs TTS avatar quality study",
        "model_recommendation": {
            "checkpoint": "artifacts/checkpoints/expanded_bgru_no_speaker_v1/best.pt",
            "face_refiner": "artifacts/refiners/expanded_bgru_no_speaker_v1_face_refiner.npz",
            "random_blinks": True,
            "blink_strength": 1.0,
        },
        "assumptions": [
            "Nikola FC files were matched exactly to spk03 source recordings.",
            "Marko (Stefan folder) and Cope files are assumed to follow the same five prompts in the same order as the Nikola set.",
        ],
        "survey_order": SURVEY_ORDER,
        "items": items,
    }
    write_text(output_root / "study_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    with (output_root / "video_upload_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "survey_position",
                "key",
                "label",
                "condition",
                "expected_mp4_name",
                "drive_file_id",
                "text",
            ],
        )
        writer.writeheader()
        by_key = {item["key"]: item for item in items}
        for position, key in enumerate(SURVEY_ORDER, start=1):
            item = by_key[key]
            writer.writerow(
                {
                    "survey_position": position,
                    "key": key,
                    "label": item["label"],
                    "condition": item["condition"],
                    "expected_mp4_name": f"{item['label']}.mp4",
                    "drive_file_id": "",
                    "text": item["text"],
                }
            )

    write_survey_files(output_root, items)
    write_bundle_readme(output_root, items)
    return items


def write_bundle_readme(output_root: Path, items: list[dict[str, str]]) -> None:
    lines = [
        "# Three-Speaker User Study Bundle",
        "",
        "Ovaj bundle je pripremljen za mobilno ocenjivanje kvaliteta avatar animacije nad tri uslova:",
        "",
        "- `Nikola`: vidjeni govornik (`seen`)",
        "- `Marko`: nevidjeni govornik (`unseen`)",
        "- `Cope`: TTS (`tts`)",
        "",
        "## Sta je unutra",
        "",
        "- `audio_inputs/`: 15 WAV fajlova preimenovanih za studiju",
        "- `text_inputs/`: 15 TXT fajlova sa tekstom za inference",
        "- `predictions/`: generisani CSV + meta nakon inference",
        "- `avatar_ready/`: upareni WAV + CSV fajlovi za FTNFacialRig",
        "- `google_apps_script/Code.gs`: mobilni survey template za Google Apps Script",
        "- `video_upload_manifest.csv`: spisak videa i mesta gde treba upisati Google Drive ID",
        "",
        "## Vazna napomena",
        "",
        "Tekst za `Nikola` set je povucen tacno iz `spk03` transkripta jer su FC fajlovi identifikovani kao originalni snimci tog govornika.",
        "Tekst za `Marko` i `Cope` set je trenutno postavljen pod pretpostavkom da pratite istih 5 recenica u istom redosledu kao `Nikola`.",
        "Ako to nije tacno, samo izmeni odgovarajuce `.txt` fajlove u `text_inputs/` i ponovo pokreni inference.",
        "",
        "## Kako dalje",
        "",
        "1. Otvori FTNFacialRig i za svaki fajl iz `avatar_ready/` snimi MP4 ekran.",
        "2. Uploaduj 15 MP4 fajlova na Google Drive.",
        "3. Upisi njihove `drive_file_id` vrednosti u `video_upload_manifest.csv` ili direktno u `google_apps_script/Code.gs`.",
        "4. Deployuj Apps Script kao web app i posalji link ispitanicima.",
        "",
        "Ako promenis neke `.txt` fajlove, ponovo pokreni inference ovom komandom:",
        "",
        "```powershell",
        "python scripts/infer_folder.py --checkpoint artifacts/checkpoints/expanded_bgru_no_speaker_v1/best.pt --face-refiner artifacts/refiners/expanded_bgru_no_speaker_v1_face_refiner.npz --input-dir artifacts/user_study/three_speakers_june_2026/audio_inputs --text-dir artifacts/user_study/three_speakers_june_2026/text_inputs --output-dir artifacts/user_study/three_speakers_june_2026/predictions --device cpu --default-speaker spk03 --random-blinks --blink-strength 1.0",
        "```",
        "",
        "## Stavke u studiji",
        "",
    ]
    for item in items:
        lines.append(f"- `{item['label']}` -> `{item['key']}.wav/.csv` ({item['condition']})")

    write_text(output_root / "README.md", "\n".join(lines) + "\n")


def build_code_gs(items: list[dict[str, str]]) -> str:
    ordered_items = []
    by_key = {item["key"]: item for item in items}
    for key in SURVEY_ORDER:
        item = by_key[key]
        ordered_items.append(
            {
                "key": item["key"],
                "label": item["label"],
                "condition": item["condition"],
                "driveId": "",
            }
        )

    video_items_js = json.dumps(ordered_items, ensure_ascii=False, indent=2)
    return f"""const SHEET_ID = '';
const VIDEO_ITEMS = {video_items_js};

function doGet() {{
  return HtmlService.createHtmlOutput(buildHtml_())
    .setTitle('Avatar anketa')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}}

function saveScore(data) {{
  const ss = SpreadsheetApp.openById(SHEET_ID);
  const sheet = ss.getSheets()[0];
  if (sheet.getLastRow() === 0) {{
    sheet.appendRow(['Timestamp', 'Participant', 'VideoLabel', 'Condition', 'Score']);
  }}
  sheet.appendRow([
    new Date(),
    data.participant,
    data.videoLabel,
    data.condition,
    data.score
  ]);
  return {{ ok: true }};
}}

function buildHtml_() {{
  return `<!DOCTYPE html>
<html lang="sr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Avatar anketa</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0d1020;
      color: #f5f7fb;
      padding: 20px 16px 40px;
      min-height: 100vh;
    }}
    .wrap {{ max-width: 720px; margin: 0 auto; }}
    .eyebrow {{
      text-align: center;
      text-transform: uppercase;
      letter-spacing: 0.22em;
      font-size: 12px;
      font-weight: 700;
      color: #8ea2ff;
      margin-bottom: 12px;
    }}
    h1 {{
      text-align: center;
      font-family: Georgia, serif;
      font-size: 32px;
      line-height: 1.2;
      margin-bottom: 12px;
    }}
    .sub {{
      text-align: center;
      color: #b7bfdc;
      line-height: 1.6;
      margin-bottom: 24px;
    }}
    .card {{
      background: #171b2f;
      border: 1px solid #29304f;
      border-radius: 24px;
      padding: 20px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
    }}
    .title {{
      font-family: Georgia, serif;
      font-size: 24px;
      margin-bottom: 8px;
    }}
    .muted {{
      color: #a9b2d0;
      line-height: 1.6;
      margin-bottom: 18px;
    }}
    label {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-weight: 700;
      color: #95a0c8;
      margin-bottom: 8px;
    }}
    input {{
      width: 100%;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid #374067;
      background: #0f1326;
      color: #f5f7fb;
      font-size: 16px;
      outline: none;
      margin-bottom: 12px;
    }}
    input.err {{ border-color: #ff7b9c; }}
    .errtxt {{
      display: none;
      color: #ff9ab0;
      font-size: 14px;
      margin-bottom: 12px;
    }}
    .errtxt.show {{ display: block; }}
    .infobox {{
      background: rgba(142, 162, 255, 0.08);
      border: 1px solid rgba(142, 162, 255, 0.22);
      border-radius: 16px;
      padding: 14px 16px;
      line-height: 1.6;
      color: #ced5ef;
      margin-bottom: 18px;
    }}
    .primary {{
      width: 100%;
      border: none;
      border-radius: 18px;
      padding: 16px 18px;
      font-size: 17px;
      font-weight: 700;
      cursor: pointer;
      color: white;
      background: linear-gradient(135deg, #7e6dff, #9f8cff);
    }}
    .primary.off {{ opacity: 0.35; pointer-events: none; }}
    #survey, #thanks {{ display: none; }}
    #survey.show, #thanks.show {{ display: block; }}
    .progress {{
      margin-bottom: 18px;
    }}
    .progressTop {{
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      margin-bottom: 8px;
      color: #a9b2d0;
    }}
    .track {{
      height: 10px;
      background: #232944;
      border-radius: 999px;
      overflow: hidden;
    }}
    .fill {{
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: linear-gradient(90deg, #7e6dff, #ff7fb7);
      transition: width 0.3s ease;
    }}
    .pill {{
      display: inline-block;
      margin-bottom: 14px;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid rgba(142, 162, 255, 0.22);
      background: rgba(142, 162, 255, 0.08);
      color: #8ea2ff;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}
    .videoBox {{
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 18px;
      border: 1px solid #2f3760;
      overflow: hidden;
      margin-bottom: 18px;
      text-decoration: none;
      display: block;
      background: linear-gradient(135deg, #0f1326, #1a2140);
      position: relative;
    }}
    .videoInner {{
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      padding: 18px;
      text-align: center;
    }}
    .playCircle {{
      width: 72px;
      height: 72px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg, #7e6dff, #9f8cff);
      box-shadow: 0 10px 30px rgba(126, 109, 255, 0.4);
    }}
    .playLabel {{ font-size: 18px; font-weight: 700; }}
    .playSub {{ font-size: 14px; color: #a9b2d0; line-height: 1.5; }}
    .question {{
      font-family: Georgia, serif;
      font-size: 21px;
      line-height: 1.5;
      margin-bottom: 18px;
    }}
    .scaleLegend {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      font-size: 13px;
      color: #a9b2d0;
      line-height: 1.5;
      margin-bottom: 12px;
    }}
    .scaleLegend span:last-child {{ text-align: right; }}
    .scoreRow {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
      margin-bottom: 18px;
    }}
    .scoreBtn {{
      min-height: 62px;
      border-radius: 16px;
      border: 1px solid #374067;
      background: #11162d;
      color: #b9c2e4;
      font-size: 24px;
      font-weight: 700;
      font-family: Georgia, serif;
      cursor: pointer;
    }}
    .scoreBtn.on {{
      color: white;
      border-color: #7e6dff;
      background: linear-gradient(135deg, #7e6dff, #9f8cff);
      box-shadow: 0 10px 24px rgba(126, 109, 255, 0.32);
    }}
    .thanks {{
      text-align: center;
      padding: 40px 12px;
    }}
    .thanks h2 {{
      font-family: Georgia, serif;
      font-size: 30px;
      margin-bottom: 12px;
    }}
    .thanks p {{
      color: #b7bfdc;
      line-height: 1.7;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="eyebrow">Istraživanje</div>
    <h1>Procena kvaliteta animacije avatara</h1>
    <p class="sub">Pogledajte svaki video i ocenite koliko prirodno deluje način na koji avatar izgovara tekst.</p>

    <div class="card" id="intro">
      <div class="title">Dobrodošli</div>
      <div class="muted">Anketa sadrži 15 kratkih video snimaka. Za svaki video dodeljuje se ocena od 1 do 5, gde je 1 potpuno nerealistično, a 5 potpuno realistično.</div>
      <label for="participantName">Ime i prezime *</label>
      <input id="participantName" type="text" placeholder="npr. Ana Petrović" autocomplete="name" />
      <div class="errtxt" id="nameError">Molimo unesite ime i prezime pre početka.</div>
      <div class="infobox">Video se otvara u novom tabu preko Google Drive linka. Nakon gledanja vratite se u anketu i unesite ocenu.</div>
      <button class="primary" onclick="startSurvey()">Započni anketu</button>
    </div>

    <div id="survey">
      <div class="progress">
        <div class="progressTop">
          <span>Napredak</span>
          <span id="progressLabel">1 / 15</span>
        </div>
        <div class="track"><div class="fill" id="progressFill"></div></div>
      </div>

      <div class="card" id="questionCard">
        <div class="pill" id="videoTag">Video 01</div>
        <a class="videoBox" id="videoLink" href="#" target="_blank" rel="noopener">
          <div class="videoInner">
            <div class="playCircle">
              <svg viewBox="0 0 24 24" width="28" height="28" fill="white"><path d="M8 5v14l11-7z"></path></svg>
            </div>
            <div class="playLabel">Tapnite da otvorite video</div>
            <div class="playSub" id="videoSub">Google Drive pregled za izabrani snimak</div>
          </div>
        </a>
        <div class="question">Koliko prirodno deluje animacija avatara u ovom snimku?</div>
        <div class="scaleLegend">
          <span>1 – Potpuno nerealistično</span>
          <span>5 – Potpuno realistično</span>
        </div>
        <div class="scoreRow">
          <button class="scoreBtn" onclick="pickScore(1)">1</button>
          <button class="scoreBtn" onclick="pickScore(2)">2</button>
          <button class="scoreBtn" onclick="pickScore(3)">3</button>
          <button class="scoreBtn" onclick="pickScore(4)">4</button>
          <button class="scoreBtn" onclick="pickScore(5)">5</button>
        </div>
        <button class="primary off" id="nextButton" onclick="submitAndNext()">Sledeći video</button>
      </div>
    </div>

    <div class="card thanks" id="thanks">
      <h2>Hvala vam!</h2>
      <p>Vaše ocene su uspešno zabeležene. Zahvaljujemo se na učešću.</p>
    </div>
  </div>

  <script>
    const ITEMS = VIDEO_ITEMS;
    let participant = '';
    let currentIndex = 0;
    let selectedScore = null;

    function startSurvey() {{
      const input = document.getElementById('participantName');
      participant = input.value.trim();
      if (!participant) {{
        input.classList.add('err');
        document.getElementById('nameError').classList.add('show');
        input.focus();
        return;
      }}
      input.classList.remove('err');
      document.getElementById('nameError').classList.remove('show');
      document.getElementById('intro').style.display = 'none';
      document.getElementById('survey').classList.add('show');
      loadQuestion(0);
      window.scrollTo(0, 0);
    }}

    function buildDriveUrl(driveId) {{
      return driveId ? ('https://drive.google.com/file/d/' + driveId + '/view') : '#';
    }}

    function loadQuestion(index) {{
      const item = ITEMS[index];
      document.getElementById('videoTag').textContent = 'Video ' + String(index + 1).padStart(2, '0');
      document.getElementById('progressLabel').textContent = (index + 1) + ' / ' + ITEMS.length;
      document.getElementById('progressFill').style.width = ((index / ITEMS.length) * 100) + '%';

      const videoLink = document.getElementById('videoLink');
      videoLink.href = buildDriveUrl(item.driveId);
      document.getElementById('videoSub').textContent = item.label + ' (' + item.condition + ')';

      document.querySelectorAll('.scoreBtn').forEach(btn => btn.classList.remove('on'));
      document.getElementById('nextButton').classList.add('off');
      document.getElementById('nextButton').textContent = index === ITEMS.length - 1 ? 'Završi anketu' : 'Sledeći video';
      selectedScore = null;
    }}

    function pickScore(score) {{
      selectedScore = score;
      document.querySelectorAll('.scoreBtn').forEach((btn, idx) => {{
        btn.classList.toggle('on', idx + 1 === score);
      }});
      document.getElementById('nextButton').classList.remove('off');
    }}

    function submitAndNext() {{
      if (selectedScore === null) {{
        return;
      }}
      const item = ITEMS[currentIndex];
      google.script.run.saveScore({{
        participant: participant,
        videoLabel: item.label,
        condition: item.condition,
        score: selectedScore
      }});

      if (currentIndex === ITEMS.length - 1) {{
        document.getElementById('survey').style.display = 'none';
        document.getElementById('thanks').classList.add('show');
        return;
      }}

      currentIndex += 1;
      loadQuestion(currentIndex);
      window.scrollTo(0, 0);
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      const input = document.getElementById('participantName');
      input.addEventListener('input', () => {{
        input.classList.remove('err');
        document.getElementById('nameError').classList.remove('show');
      }});
      input.addEventListener('keydown', event => {{
        if (event.key === 'Enter') {{
          startSurvey();
        }}
      }});
    }});
  </script>
</body>
</html>`;
}}
"""


def write_survey_files(output_root: Path, items: list[dict[str, str]]) -> None:
    survey_dir = ensure_dir(output_root / "google_apps_script")
    write_text(survey_dir / "Code.gs", build_code_gs(items))
    setup_md = "\n".join(
        [
            "# Google Apps Script setup",
            "",
            "1. Napravi Google Sheet koji ce cuvati rezultate i prekopiraj njegov ID u `SHEET_ID` u `Code.gs`.",
            "2. Snimi 15 MP4 videa iz `avatar_ready/` foldera i uploaduj ih na Google Drive.",
            "3. U `VIDEO_ITEMS` za svaku stavku upisi odgovarajuci `driveId`.",
            "4. U Google Apps Script projektu ubaci sadrzaj `Code.gs` i deployuj kao Web App.",
            "5. Posalji generisani link ispitanicima i sakupi ocene.",
            "",
            "Preporuka: imenuj MP4 fajlove isto kao `label` kolonu u `video_upload_manifest.csv`.",
        ]
    )
    write_text(survey_dir / "SETUP.md", setup_md + "\n")


def run_inference(output_root: Path) -> None:
    predictions_dir = ensure_dir(output_root / "predictions")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "infer_folder.py"),
        "--checkpoint",
        str(ROOT / "artifacts" / "checkpoints" / "expanded_bgru_no_speaker_v1" / "best.pt"),
        "--face-refiner",
        str(ROOT / "artifacts" / "refiners" / "expanded_bgru_no_speaker_v1_face_refiner.npz"),
        "--input-dir",
        str(output_root / "audio_inputs"),
        "--text-dir",
        str(output_root / "text_inputs"),
        "--output-dir",
        str(predictions_dir),
        "--device",
        "cpu",
        "--default-speaker",
        "spk03",
        "--random-blinks",
        "--blink-strength",
        "1.0",
    ]
    subprocess.run(command, check=True, cwd=str(ROOT))

    avatar_ready = ensure_dir(output_root / "avatar_ready")
    for wav_path in sorted((output_root / "audio_inputs").glob("*.wav")):
        shutil.copy2(wav_path, avatar_ready / wav_path.name)
        csv_path = predictions_dir / f"{wav_path.stem}.csv"
        if csv_path.exists():
            shutil.copy2(csv_path, avatar_ready / csv_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a three-speaker mobile user study bundle.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "user_study" / "three_speakers_june_2026",
    )
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_root)
    prepare_bundle(args.output_root)
    if not args.skip_inference:
        run_inference(args.output_root)
    print(f"Prepared user study bundle at {args.output_root}")


if __name__ == "__main__":
    main()
