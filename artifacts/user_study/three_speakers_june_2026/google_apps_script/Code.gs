const SHEET_ID = '';
const VIDEO_ITEMS = [
  {
    "key": "marko_01",
    "label": "Marko1",
    "condition": "unseen",
    "driveId": ""
  },
  {
    "key": "nikola_03",
    "label": "Nikola3",
    "condition": "seen",
    "driveId": ""
  },
  {
    "key": "cope_02",
    "label": "Cope2",
    "condition": "tts",
    "driveId": ""
  },
  {
    "key": "nikola_01",
    "label": "Nikola1",
    "condition": "seen",
    "driveId": ""
  },
  {
    "key": "marko_04",
    "label": "Marko4",
    "condition": "unseen",
    "driveId": ""
  },
  {
    "key": "cope_05",
    "label": "Cope5",
    "condition": "tts",
    "driveId": ""
  },
  {
    "key": "nikola_05",
    "label": "Nikola5",
    "condition": "seen",
    "driveId": ""
  },
  {
    "key": "cope_01",
    "label": "Cope1",
    "condition": "tts",
    "driveId": ""
  },
  {
    "key": "marko_02",
    "label": "Marko2",
    "condition": "unseen",
    "driveId": ""
  },
  {
    "key": "nikola_02",
    "label": "Nikola2",
    "condition": "seen",
    "driveId": ""
  },
  {
    "key": "cope_04",
    "label": "Cope4",
    "condition": "tts",
    "driveId": ""
  },
  {
    "key": "marko_05",
    "label": "Marko5",
    "condition": "unseen",
    "driveId": ""
  },
  {
    "key": "nikola_04",
    "label": "Nikola4",
    "condition": "seen",
    "driveId": ""
  },
  {
    "key": "cope_03",
    "label": "Cope3",
    "condition": "tts",
    "driveId": ""
  },
  {
    "key": "marko_03",
    "label": "Marko3",
    "condition": "unseen",
    "driveId": ""
  }
];

function doGet() {
  return HtmlService.createHtmlOutput(buildHtml_())
    .setTitle('Avatar anketa')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

function saveScore(data) {
  const ss = SpreadsheetApp.openById(SHEET_ID);
  const sheet = ss.getSheets()[0];
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(['Timestamp', 'Participant', 'VideoLabel', 'Condition', 'Score']);
  }
  sheet.appendRow([
    new Date(),
    data.participant,
    data.videoLabel,
    data.condition,
    data.score
  ]);
  return { ok: true };
}

function buildHtml_() {
  return `<!DOCTYPE html>
<html lang="sr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Avatar anketa</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0d1020;
      color: #f5f7fb;
      padding: 20px 16px 40px;
      min-height: 100vh;
    }
    .wrap { max-width: 720px; margin: 0 auto; }
    .eyebrow {
      text-align: center;
      text-transform: uppercase;
      letter-spacing: 0.22em;
      font-size: 12px;
      font-weight: 700;
      color: #8ea2ff;
      margin-bottom: 12px;
    }
    h1 {
      text-align: center;
      font-family: Georgia, serif;
      font-size: 32px;
      line-height: 1.2;
      margin-bottom: 12px;
    }
    .sub {
      text-align: center;
      color: #b7bfdc;
      line-height: 1.6;
      margin-bottom: 24px;
    }
    .card {
      background: #171b2f;
      border: 1px solid #29304f;
      border-radius: 24px;
      padding: 20px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
    }
    .title {
      font-family: Georgia, serif;
      font-size: 24px;
      margin-bottom: 8px;
    }
    .muted {
      color: #a9b2d0;
      line-height: 1.6;
      margin-bottom: 18px;
    }
    label {
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-weight: 700;
      color: #95a0c8;
      margin-bottom: 8px;
    }
    input {
      width: 100%;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid #374067;
      background: #0f1326;
      color: #f5f7fb;
      font-size: 16px;
      outline: none;
      margin-bottom: 12px;
    }
    input.err { border-color: #ff7b9c; }
    .errtxt {
      display: none;
      color: #ff9ab0;
      font-size: 14px;
      margin-bottom: 12px;
    }
    .errtxt.show { display: block; }
    .infobox {
      background: rgba(142, 162, 255, 0.08);
      border: 1px solid rgba(142, 162, 255, 0.22);
      border-radius: 16px;
      padding: 14px 16px;
      line-height: 1.6;
      color: #ced5ef;
      margin-bottom: 18px;
    }
    .primary {
      width: 100%;
      border: none;
      border-radius: 18px;
      padding: 16px 18px;
      font-size: 17px;
      font-weight: 700;
      cursor: pointer;
      color: white;
      background: linear-gradient(135deg, #7e6dff, #9f8cff);
    }
    .primary.off { opacity: 0.35; pointer-events: none; }
    #survey, #thanks { display: none; }
    #survey.show, #thanks.show { display: block; }
    .progress {
      margin-bottom: 18px;
    }
    .progressTop {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      margin-bottom: 8px;
      color: #a9b2d0;
    }
    .track {
      height: 10px;
      background: #232944;
      border-radius: 999px;
      overflow: hidden;
    }
    .fill {
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: linear-gradient(90deg, #7e6dff, #ff7fb7);
      transition: width 0.3s ease;
    }
    .pill {
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
    }
    .videoBox {
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
    }
    .videoInner {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      padding: 18px;
      text-align: center;
    }
    .playCircle {
      width: 72px;
      height: 72px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg, #7e6dff, #9f8cff);
      box-shadow: 0 10px 30px rgba(126, 109, 255, 0.4);
    }
    .playLabel { font-size: 18px; font-weight: 700; }
    .playSub { font-size: 14px; color: #a9b2d0; line-height: 1.5; }
    .question {
      font-family: Georgia, serif;
      font-size: 21px;
      line-height: 1.5;
      margin-bottom: 18px;
    }
    .scaleLegend {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      font-size: 13px;
      color: #a9b2d0;
      line-height: 1.5;
      margin-bottom: 12px;
    }
    .scaleLegend span:last-child { text-align: right; }
    .scoreRow {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
      margin-bottom: 18px;
    }
    .scoreBtn {
      min-height: 62px;
      border-radius: 16px;
      border: 1px solid #374067;
      background: #11162d;
      color: #b9c2e4;
      font-size: 24px;
      font-weight: 700;
      font-family: Georgia, serif;
      cursor: pointer;
    }
    .scoreBtn.on {
      color: white;
      border-color: #7e6dff;
      background: linear-gradient(135deg, #7e6dff, #9f8cff);
      box-shadow: 0 10px 24px rgba(126, 109, 255, 0.32);
    }
    .thanks {
      text-align: center;
      padding: 40px 12px;
    }
    .thanks h2 {
      font-family: Georgia, serif;
      font-size: 30px;
      margin-bottom: 12px;
    }
    .thanks p {
      color: #b7bfdc;
      line-height: 1.7;
    }
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

    function startSurvey() {
      const input = document.getElementById('participantName');
      participant = input.value.trim();
      if (!participant) {
        input.classList.add('err');
        document.getElementById('nameError').classList.add('show');
        input.focus();
        return;
      }
      input.classList.remove('err');
      document.getElementById('nameError').classList.remove('show');
      document.getElementById('intro').style.display = 'none';
      document.getElementById('survey').classList.add('show');
      loadQuestion(0);
      window.scrollTo(0, 0);
    }

    function buildDriveUrl(driveId) {
      return driveId ? ('https://drive.google.com/file/d/' + driveId + '/view') : '#';
    }

    function loadQuestion(index) {
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
    }

    function pickScore(score) {
      selectedScore = score;
      document.querySelectorAll('.scoreBtn').forEach((btn, idx) => {
        btn.classList.toggle('on', idx + 1 === score);
      });
      document.getElementById('nextButton').classList.remove('off');
    }

    function submitAndNext() {
      if (selectedScore === null) {
        return;
      }
      const item = ITEMS[currentIndex];
      google.script.run.saveScore({
        participant: participant,
        videoLabel: item.label,
        condition: item.condition,
        score: selectedScore
      });

      if (currentIndex === ITEMS.length - 1) {
        document.getElementById('survey').style.display = 'none';
        document.getElementById('thanks').classList.add('show');
        return;
      }

      currentIndex += 1;
      loadQuestion(currentIndex);
      window.scrollTo(0, 0);
    }

    document.addEventListener('DOMContentLoaded', () => {
      const input = document.getElementById('participantName');
      input.addEventListener('input', () => {
        input.classList.remove('err');
        document.getElementById('nameError').classList.remove('show');
      });
      input.addEventListener('keydown', event => {
        if (event.key === 'Enter') {
          startSurvey();
        }
      });
    });
  </script>
</body>
</html>`;
}
