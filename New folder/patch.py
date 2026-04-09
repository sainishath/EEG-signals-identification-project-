import re
import sys

def main():
    file_path = r"d:\desktop\project file\New folder\withlogin.html"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. CSS changes
    css_to_add = """
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

/* ── CWT image card ── */
.cwt-card { display: none; }
.cwt-card.visible { display: block; }
.cwt-img {
  width: 100%; border-radius: 6px; display: block;
  border: 1px solid var(--border); margin-top: 4px;
}
.cwt-meta {
  font-family: var(--mono); font-size: 10px;
  color: var(--muted); margin-top: 8px;
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}
.cwt-meta span { color: var(--accent); }

/* ── Session metrics ── */
.metrics-session {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 16px; margin-top: 8px;
}
.metric-spark-label {
  font-family: var(--mono); font-size: 10px;
  color: var(--muted); letter-spacing: .12em;
  text-transform: uppercase; margin-bottom: 6px;
}
.metric-spark-val {
  font-family: var(--mono); font-size: 18px;
  font-weight: 700; margin-bottom: 6px;
}
.spark-canvas {
  width: 100%; height: 48px; display: block;
  border-radius: 4px; background: rgba(0,0,0,.3);
}

/* ── File upload (EDF) ── */
.edf-loaded {
  display: flex; align-items: center; gap: 10px;
  background: rgba(0,200,180,.06);
  border: 1px solid var(--border2);
  border-radius: 8px; padding: 12px 14px;
  margin-top: 12px;
}
.edf-icon { font-size: 20px; }
.edf-name { font-family: var(--mono); font-size: 12px; color: var(--accent); }
.edf-ch   { font-size: 11px; color: var(--muted); margin-top: 2px; }

@media (max-width: 900px) {"""

    # Replace CSS tail
    content = content.replace("::-webkit-scrollbar { width: 6px; }\n::-webkit-scrollbar-track { background: var(--bg); }\n::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }\n\n@media (max-width: 900px) {", css_to_add)

    # 2. Main left panel
    # We replace from <div class="card-label">Input Signal</div> to <button class="analyze-btn" id="analyzeBtn" onclick="analyzeSignal()" disabled>
    left_panel_old = """      <div class="card">
        <div class="card-label">Input Signal</div>
        <div class="card-title">Load EEG Data</div>
        <p style="font-size:13px;color:var(--muted);margin:8px 0 16px;">Select a demo signal or upload your own EEG file (.txt / .csv)</p>
        <div class="demo-buttons">
          <button class="demo-btn btn-normal" onclick="loadDemo('normal')">
            <div><div class="btn-label">Normal EEG</div><div class="btn-desc">Healthy background activity (Set B)</div></div>
            <span class="btn-dot"></span>
          </button>
          <button class="demo-btn btn-preictal" onclick="loadDemo('preictal')">
            <div><div class="btn-label">Preictal EEG</div><div class="btn-desc">Pre-seizure state (Set D)</div></div>
            <span class="btn-dot"></span>
          </button>
          <button class="demo-btn btn-seizure" onclick="loadDemo('seizure')">
            <div><div class="btn-label">Seizure EEG</div><div class="btn-desc">Ictal activity (Set E)</div></div>
            <span class="btn-dot"></span>
          </button>
        </div>
        <label class="upload-zone" for="fileInput">
          <input type="file" id="fileInput" accept=".txt,.csv,.edf" onchange="handleFileUpload(event)">
          <div class="upload-icon">📂</div>
          <div class="upload-text">Drop your EEG file here<br><span style="color:var(--accent)">or click to browse</span></div>
        </label>
        <button class="analyze-btn" id="analyzeBtn" onclick="analyzeSignal()" disabled>
          <span>RUN ANALYSIS</span>
        </button>
      </div>"""

    left_panel_new = """      <div class="card">
        <div class="card-label">Input Signal</div>
        <div class="card-title">Load EEG Data</div>
        <p style="font-size:13px;color:var(--muted);margin:8px 0 16px;">Select a demo signal or upload your own <b style="color:var(--accent)">.edf</b> file (16-22 channels)</p>
        <div class="demo-buttons">
          <button class="demo-btn btn-normal" onclick="loadDemo('normal', event)">
            <div><div class="btn-label">Normal EEG</div><div class="btn-desc">Healthy background activity</div></div>
            <span class="btn-dot"></span>
          </button>
          <button class="demo-btn btn-preictal" onclick="loadDemo('preictal', event)">
            <div><div class="btn-label">Preictal EEG</div><div class="btn-desc">Pre-seizure state</div></div>
            <span class="btn-dot"></span>
          </button>
          <button class="demo-btn btn-seizure" onclick="loadDemo('seizure', event)">
            <div><div class="btn-label">Seizure EEG</div><div class="btn-desc">Ictal activity</div></div>
            <span class="btn-dot"></span>
          </button>
        </div>
        <label class="upload-zone" for="fileInput" id="uploadZoneLabel">
          <input type="file" id="fileInput" accept=".edf" onchange="handleFileUpload(event)">
          <div class="upload-icon">📂</div>
          <div class="upload-text">Upload EDF file<br><span style="color:var(--accent)">click to browse (.edf only)</span></div>
        </label>
        <!-- EDF loaded indicator (hidden by default) -->
        <div class="edf-loaded" id="edfLoaded" style="display:none;">
          <span class="edf-icon">🧠</span>
          <div>
            <div class="edf-name" id="edfName">—</div>
            <div class="edf-ch" id="edfChannelInfo">Upload to see channel info</div>
          </div>
        </div>
        <button class="analyze-btn" id="analyzeBtn" onclick="analyzeSignal()" disabled>
          <span id="analyzeBtnText">RUN ANALYSIS</span>
        </button>
      </div>"""

    content = content.replace(left_panel_old, left_panel_new)

    # 3. Add CWT Scalogram card
    eeg_canvas_old = """      <div class="card">
        <div class="card-label">EEG Waveform</div>
        <canvas id="eegCanvas" width="900" height="160"></canvas>
        <div class="waveform-controls">
          <span class="wc-label">SAMPLES:</span><span class="wc-val" id="sampleCount">—</span>
          <span style="margin:0 8px;color:var(--border2)">|</span>
          <span class="wc-label">DURATION:</span><span class="wc-val" id="sigDuration">—</span>
          <span style="margin:0 8px;color:var(--border2)">|</span>
          <span class="wc-label">AMPLITUDE:</span><span class="wc-val" id="sigAmp">—</span>
        </div>
      </div>"""

    eeg_canvas_new = eeg_canvas_old + """

      <!-- CWT Scalogram card — shown after analysis -->
      <div class="card cwt-card" id="cwtCard">
        <div class="card-label">CWT Scalogram</div>
        <img class="cwt-img" id="cwtImage" src="" alt="CWT scalogram" loading="lazy">
        <div class="cwt-meta">
          Wavelet: <span>cmor1.5-1.0</span>
          &nbsp;·&nbsp; Freq range: <span>1–50 Hz</span>
          &nbsp;·&nbsp; Channel: <span id="cwtChInfo">—</span>
        </div>
      </div>"""
    
    content = content.replace(eeg_canvas_old, eeg_canvas_new)

    # 4. Add Metric Card
    history_table_old = """      <div class="card">
        <div class="card-label">Analysis History</div>
        <div id="historyEmpty" class="empty-state" style="padding:24px;">
          <div class="empty-text" style="font-size:12px;">No analyses run yet</div>
        </div>
        <table class="history-table" id="historyTable" style="display:none;">
          <thead>
            <tr><th>#</th><th>Time</th><th>Signal Type</th><th>Prediction</th><th>Confidence</th></tr>
          </thead>
          <tbody id="historyBody"></tbody>
        </table>
      </div>"""

    history_table_new = """      <!-- Session Metrics card -->
      <div class="card" id="sessionMetricsCard" style="display:none;">
        <div class="card-label">Session Metrics</div>
        <div class="metrics-session">
          <div>
            <div class="metric-spark-label">Loss (NLL proxy)</div>
            <div class="metric-spark-val" id="lossVal" style="color:var(--preictal);">—</div>
            <canvas class="spark-canvas" id="lossChart"></canvas>
          </div>
          <div>
            <div class="metric-spark-label">Seizure Sensitivity</div>
            <div class="metric-spark-val" id="sensitivityVal" style="color:var(--seizure);">—</div>
            <canvas class="spark-canvas" id="sensitivityChart"></canvas>
          </div>
        </div>
        <p style="font-size:11px;color:var(--muted);margin-top:10px;font-family:var(--mono);">Metrics update per analysis run in this session.</p>
      </div>

      <div class="card">
        <div class="card-label">Analysis History</div>
        <div id="historyEmpty" class="empty-state" style="padding:24px;">
          <div class="empty-text" style="font-size:12px;">No analyses run yet</div>
        </div>
        <table class="history-table" id="historyTable" style="display:none;">
          <thead>
            <tr><th>#</th><th>Time</th><th>Source</th><th>Channels</th><th>Prediction</th><th>Confidence</th></tr>
          </thead>
          <tbody id="historyBody"></tbody>
        </table>
      </div>"""

    content = content.replace(history_table_old, history_table_new)


    # 5. Replace JS
    js_old = content[content.find("<script>"):content.find("</body>")]
    
    js_new = """<script>
/* ═══════════════════════════════════════════════════════════
   USER STORE  (localStorage-backed)
══════════════════════════════════════════════════════════ */
function getUsers()  { return JSON.parse(localStorage.getItem('ns_users')  || '[]');   }
function saveUsers(u){ localStorage.setItem('ns_users',  JSON.stringify(u)); }
function getSession(){ return JSON.parse(localStorage.getItem('ns_session') || 'null'); }
function saveSession(u){ localStorage.setItem('ns_session', JSON.stringify(u)); }
function clearSession(){ localStorage.removeItem('ns_session'); }

/* seed demo account */
(function seedDemo() {
  const users = getUsers();
  if (!users.find(u => u.email === 'demo@neuroscan.ai')) {
    users.push({ firstName:'Demo', lastName:'User', email:'demo@neuroscan.ai', password:'demo1234', role:'Researcher' });
    saveUsers(users);
  }
})();

/* ═══════════════════════════════════════════════════════════
   AUTH — TAB SWITCH
══════════════════════════════════════════════════════════ */
function switchTab(tab) {
  document.querySelectorAll('.auth-tab').forEach((t,i) =>
    t.classList.toggle('active', (i===0 && tab==='login') || (i===1 && tab==='register'))
  );
  document.getElementById('loginForm').classList.toggle('active',    tab==='login');
  document.getElementById('registerForm').classList.toggle('active', tab==='register');
  clearErrors();
}
function clearErrors() {
  document.querySelectorAll('.field-hint.err').forEach(e => e.classList.remove('show'));
  document.querySelectorAll('.field input').forEach(i => i.classList.remove('error'));
}

/* ═══════════════════════════════════════════════════════════
   TOAST
══════════════════════════════════════════════════════════ */
function showToast(msg, type = 'ok') {
  const t = document.getElementById('toast');
  const colors = { ok:'#22c55e', err:'#ef4444', warn:'#f59e0b', info:'#00c8b4' };
  document.getElementById('toastDot').style.background = colors[type] || colors.info;
  document.getElementById('toastMsg').textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3200);
}

/* ═══════════════════════════════════════════════════════════
   PASSWORD STRENGTH
══════════════════════════════════════════════════════════ */
function checkStrength(pw) {
  const bars   = [1,2,3,4].map(i => document.getElementById('pb'+i));
  const hint   = document.getElementById('pwHint');
  bars.forEach(b => { b.className = 'pw-bar'; });
  if (!pw.length) { hint.textContent = 'Minimum 8 characters'; hint.style.color = 'var(--muted)'; return; }
  let score = 0;
  if (pw.length >= 8)          score++;
  if (/[A-Z]/.test(pw))       score++;
  if (/[0-9]/.test(pw))       score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;
  const cls = ['s1','s2','s3','s4'];
  for (let i=0;i<score;i++) bars[i].classList.add(cls[score-1]);
  const labels  = ['','Weak','Fair','Good','Strong'];
  const hColors = ['','var(--seizure)','var(--preictal)','var(--accent)','var(--normal)'];
  hint.textContent  = labels[score];
  hint.style.color  = hColors[score];
}

/* ═══════════════════════════════════════════════════════════
   LOGIN
══════════════════════════════════════════════════════════ */
function handleLogin(e) {
  e.preventDefault(); clearErrors();
  const email = document.getElementById('loginEmail').value.trim();
  const pass  = document.getElementById('loginPass').value;
  const btn   = document.getElementById('loginBtn');
  btn.disabled = true; btn.classList.add('loading');
  setTimeout(() => {
    const user = getUsers().find(u => u.email === email);
    if (!user) {
      document.getElementById('loginEmail').classList.add('error');
      document.getElementById('loginEmailErr').classList.add('show');
      btn.disabled=false; btn.classList.remove('loading'); return;
    }
    if (user.password !== pass) {
      document.getElementById('loginPass').classList.add('error');
      document.getElementById('loginPassErr').classList.add('show');
      btn.disabled=false; btn.classList.remove('loading'); return;
    }
    if (document.getElementById('rememberMe').checked) saveSession(user);
    enterApp(user);
    btn.disabled=false; btn.classList.remove('loading');
  }, 800);
}
function demoLogin() { enterApp(getUsers().find(u => u.email === 'demo@neuroscan.ai')); }
function showForgot() { showToast('Password reset not available in demo mode.','warn'); }

/* ═══════════════════════════════════════════════════════════
   REGISTER
══════════════════════════════════════════════════════════ */
function handleRegister(e) {
  e.preventDefault(); clearErrors();
  const first = document.getElementById('regFirst').value.trim();
  const last  = document.getElementById('regLast').value.trim();
  const email = document.getElementById('regEmail').value.trim();
  const role  = document.getElementById('regRole').value.trim();
  const pass  = document.getElementById('regPass').value;
  const pass2 = document.getElementById('regPass2').value;
  const btn   = document.getElementById('registerBtn');
  if (pass !== pass2) {
    document.getElementById('regPass2').classList.add('error');
    document.getElementById('regPassErr').classList.add('show');
    return;
  }
  btn.disabled=true; btn.classList.add('loading');
  setTimeout(() => {
    const users = getUsers();
    if (users.find(u => u.email === email)) {
      document.getElementById('regEmail').classList.add('error');
      document.getElementById('regEmailErr').classList.add('show');
      btn.disabled=false; btn.classList.remove('loading'); return;
    }
    const newUser = { firstName:first, lastName:last, email, password:pass, role:role||'User' };
    users.push(newUser); saveUsers(users);
    showToast('Account created! Signing you in…','ok');
    setTimeout(() => enterApp(newUser), 800);
    btn.disabled=false; btn.classList.remove('loading');
  }, 900);
}

/* ═══════════════════════════════════════════════════════════
   ENTER / LEAVE APP
══════════════════════════════════════════════════════════ */
function enterApp(user) {
  const loginPage = document.getElementById('loginPage');
  loginPage.classList.add('hiding');
  document.getElementById('userNameDisplay').textContent = user.firstName + ' ' + user.lastName;
  document.getElementById('userAvatar').textContent = user.firstName[0].toUpperCase();
  setTimeout(() => {
    loginPage.style.display = 'none';
    document.getElementById('appPage').classList.add('visible');
    drawWaveform(null);
  }, 480);
}
function logout() {
  clearSession();
  document.getElementById('appPage').classList.remove('visible');
  const lp = document.getElementById('loginPage');
  lp.style.display='flex'; lp.classList.remove('hiding');
  clearResult();
  currentSignal = null; currentEdfFile = null;
  document.querySelectorAll('.demo-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('edfLoaded').style.display = 'none';
  document.getElementById('uploadZoneLabel').style.display = '';
  showToast('You have been signed out.','info');
}

/* ═══════════════════════════════════════════════════════════
   AUTO-LOGIN + INIT
══════════════════════════════════════════════════════════ */
window.addEventListener('DOMContentLoaded', () => {
  const session = getSession();
  if (session) enterApp(session);
  animateDecoCanvas();
  animateBgWave();
});

/* ═══════════════════════════════════════════════════════════
   DECORATIVE EEG CANVAS (login page)
══════════════════════════════════════════════════════════ */
let decoOffset = 0;
function animateDecoCanvas() {
  const c = document.getElementById('decoCanvas');
  if (!c) return;
  const ctx = c.getContext('2d'), W=c.width, H=c.height;
  (function frame() {
    ctx.clearRect(0,0,W,H);
    ctx.strokeStyle='#00c8b4'; ctx.lineWidth=1.5; ctx.lineJoin='round';
    ctx.beginPath();
    for (let x=0;x<W;x++) {
      const t=(x+decoOffset)/W*4*Math.PI;
      const y=H/2+Math.sin(t)*12+Math.sin(t*2.3+1)*6+Math.sin(t*4.7+2)*3;
      x===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.stroke(); decoOffset+=1.5; requestAnimationFrame(frame);
  })();
}
function animateBgWave() {
  const path = document.getElementById('bgWave');
  if (!path) return;
  let off = 0;
  (function frame() {
    const W=1440,H=200,pts=[];
    for (let x=0;x<=W;x+=12) {
      const y=H/2+Math.sin((x+off)/120)*40+Math.sin((x+off)/60)*15;
      pts.push(`${x},${y}`);
    }
    path.setAttribute('d','M '+pts.join(' L ')); off+=0.8; requestAnimationFrame(frame);
  })();
}

/* ═══════════════════════════════════════════════════════════
   EEG CANVAS WAVEFORM
══════════════════════════════════════════════════════════ */
function drawWaveform(signal, color='#00c8b4') {
  const canvas = document.getElementById('eegCanvas');
  if (!canvas) return;
  const ctx=canvas.getContext('2d'), W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='rgba(0,0,0,0.3)'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='rgba(255,255,255,0.05)'; ctx.lineWidth=0.5;
  for (let i=0;i<=4;i++) {
    const y=(H/4)*i;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
  }
  if (!signal || signal.length<2) return;
  const step=W/signal.length;
  const mn=Math.min(...signal), mx=Math.max(...signal), range=mx-mn||1, pad=H*0.1;
  const toY=v=>pad+((mx-v)/range)*(H-2*pad);
  ctx.shadowColor=color; ctx.shadowBlur=8;
  ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.lineJoin='round';
  ctx.beginPath();
  signal.forEach((v,i)=>{ i===0?ctx.moveTo(i*step,toY(v)):ctx.lineTo(i*step,toY(v)); });
  ctx.stroke(); ctx.shadowBlur=0;
  document.getElementById('sampleCount').textContent  = signal.length.toLocaleString();
  document.getElementById('sigDuration').textContent  = (signal.length/256).toFixed(1)+' s';
  document.getElementById('sigAmp').textContent       = (mx-mn).toFixed(1)+' µV';
}

/* ═══════════════════════════════════════════════════════════
   SPARKLINE CHART
══════════════════════════════════════════════════════════ */
function drawSparkline(canvasId, data, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !data.length) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr || 200 * dpr;
  canvas.height = 48 * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = canvas.width/dpr, H = canvas.height/dpr;
  ctx.clearRect(0,0,W,H);

  const mn=Math.min(...data), mx=Math.max(...data), range=mx-mn||0.01;
  const toY=v=>H-4-((v-mn)/range)*(H-8);
  const step=W/(Math.max(data.length-1,1));

  // gradient fill
  const grad=ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0, color+'55');
  grad.addColorStop(1, color+'00');
  ctx.beginPath();
  data.forEach((v,i)=>i===0?ctx.moveTo(i*step,toY(v)):ctx.lineTo(i*step,toY(v)));
  ctx.lineTo((data.length-1)*step,H);
  ctx.lineTo(0,H); ctx.closePath();
  ctx.fillStyle=grad; ctx.fill();

  // line
  ctx.beginPath();
  data.forEach((v,i)=>i===0?ctx.moveTo(i*step,toY(v)):ctx.lineTo(i*step,toY(v)));
  ctx.strokeStyle=color; ctx.lineWidth=1.8; ctx.lineJoin='round';
  ctx.shadowColor=color; ctx.shadowBlur=4;
  ctx.stroke(); ctx.shadowBlur=0;

  // last point dot
  const lx=(data.length-1)*step, ly=toY(data[data.length-1]);
  ctx.beginPath(); ctx.arc(lx,ly,3,0,Math.PI*2);
  ctx.fillStyle=color; ctx.fill();
}

/* ═══════════════════════════════════════════════════════════
   STATE
══════════════════════════════════════════════════════════ */
let currentSignal  = null;
let currentEdfFile = null;      // File object when EDF uploaded
let currentSource  = 'demo';    // 'demo' | 'edf'
let analysisCount  = 0;
let sessionLoss        = [];
let sessionSensitivity = [];
const API = 'http://localhost:5000/api';

/* ═══════════════════════════════════════════════════════════
   DEMO SIGNAL LOADER
══════════════════════════════════════════════════════════ */
async function loadDemo(type, evt) {
  document.querySelectorAll('.demo-btn').forEach(b => b.classList.remove('active'));
  if (evt && evt.currentTarget) evt.currentTarget.classList.add('active');
  currentEdfFile = null;
  currentSource  = 'demo';
  document.getElementById('edfLoaded').style.display      = 'none';
  document.getElementById('uploadZoneLabel').style.display = '';
  const colors = { normal:'#22c55e', preictal:'#f59e0b', seizure:'#ef4444' };
  try {
    const res = await fetch(`${API}/demo-signal/${type}`);
    if (!res.ok) throw new Error();
    currentSignal = (await res.json()).signal;
  } catch {
    currentSignal = generateLocalSignal(type);
  }
  drawWaveform(currentSignal, colors[type]);
  document.getElementById('analyzeBtn').disabled = false;
  clearResult();
}

function generateLocalSignal(type) {
  const N=512, t=Array.from({length:N},(_,i)=>i/256);
  if (type==='seizure') return t.map((ti,i)=>{ const b=(Math.floor(i/60)%2===0)?1:0.2; return Math.sin(2*Math.PI*3*ti)*800*b+Math.sin(2*Math.PI*7*ti)*300*b+(Math.random()-.5)*100; });
  if (type==='preictal') return t.map((ti,i)=>{ const r=1+i/N; return (Math.sin(2*Math.PI*4*ti)*60+Math.sin(2*Math.PI*8*ti)*30+(Math.random()-.5)*20)*r; });
  return t.map(ti=>Math.sin(2*Math.PI*10*ti)*20+Math.sin(2*Math.PI*20*ti)*10+(Math.random()-.5)*15);
}

/* ═══════════════════════════════════════════════════════════
   EDF UPLOAD
══════════════════════════════════════════════════════════ */
function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  if (!file.name.toLowerCase().endsWith('.edf')) {
    showToast('Only .edf files are accepted.', 'err');
    event.target.value = '';
    return;
  }

  currentEdfFile = file;
  currentSignal  = null;
  currentSource  = 'edf';
  document.querySelectorAll('.demo-btn').forEach(b => b.classList.remove('active'));

  // Show EDF loaded badge
  document.getElementById('edfName').textContent       = file.name;
  document.getElementById('edfChannelInfo').textContent = 'Channels will be read from file (16–22 max)';
  document.getElementById('edfLoaded').style.display      = 'flex';
  document.getElementById('uploadZoneLabel').style.display = 'none';

  // Draw placeholder waveform
  const placeholder = Array.from({length:512},(_,i)=>Math.sin(i*0.05)*30+(Math.random()-.5)*10);
  drawWaveform(placeholder, '#6b8a96');
  document.getElementById('sampleCount').textContent  = 'Loading…';
  document.getElementById('sigDuration').textContent  = '—';
  document.getElementById('sigAmp').textContent       = '—';

  document.getElementById('analyzeBtn').disabled = false;
  clearResult();
  showToast('EDF file loaded — click Run Analysis', 'info');
}

/* ═══════════════════════════════════════════════════════════
   ANALYZE SIGNAL
══════════════════════════════════════════════════════════ */
async function analyzeSignal() {
  if (!currentSignal && !currentEdfFile) return;

  const btn = document.getElementById('analyzeBtn');
  const btnTxt = document.getElementById('analyzeBtnText');
  btn.disabled=true; btn.classList.add('loading');
  btnTxt.textContent='ANALYZING…';

  try {
    let res;

    if (currentEdfFile) {
      // ── EDF upload path ──
      const fd = new FormData();
      fd.append('edf_file', currentEdfFile);
      res = await fetch(`${API}/predict`, { method:'POST', body: fd });
    } else {
      // ── Demo JSON path ──
      res = await fetch(`${API}/predict`, {
        method:'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ signal: currentSignal })
      });
    }

    const data = await res.json();
    if (!res.ok) {
      showToast(data.error || 'Server error', 'err');
      return;
    }

    // Update waveform from server's processed signal
    if (data.display_signal && data.display_signal.length) {
      const col = data.predicted_label==='Seizure'?'#ef4444'
                : data.predicted_label==='Preictal'?'#f59e0b':'#22c55e';
      drawWaveform(data.display_signal, col);
      currentSignal = data.display_signal;
    }

    // Update EDF channel info
    if (currentEdfFile && data.num_channels) {
      document.getElementById('edfChannelInfo').textContent =
        `${data.num_channels} channel(s) used · padded to 22 for model`;
    }

    showResult(data);

  } catch (err) {
    showToast('Cannot reach server — check that app.py is running on port 5000.', 'err');
    console.error(err);
  } finally {
    btn.classList.remove('loading');
    btnTxt.textContent='RUN ANALYSIS';
    btn.disabled=false;
  }
}

/* ═══════════════════════════════════════════════════════════
   SHOW RESULT
══════════════════════════════════════════════════════════ */
function showResult(data) {
  const label = data.predicted_label;
  const conf  = parseFloat(data.confidence);
  const probs = data.probabilities;
  const icons = { Normal:'✅', Preictal:'⚠️', Seizure:'🚨' };
  const subs  = {
    Normal:   'No epileptic activity detected',
    Preictal: 'Pre-seizure state — monitor closely',
    Seizure:  'Active seizure detected — alert!'
  };

  const rd = document.getElementById('resultDisplay');
  rd.className = 'result-display result-'+label.toLowerCase();
  document.getElementById('emptyState').style.display   = 'none';
  document.getElementById('resultContent').style.display = 'block';
  document.getElementById('resultIcon').textContent   = icons[label];
  document.getElementById('resultLabel').textContent  = label.toUpperCase();
  document.getElementById('resultSub').textContent    = subs[label];
  document.getElementById('confidenceVal').textContent = conf.toFixed(1)+'%';

  setTimeout(() => {
    document.getElementById('barNormal').style.width   = probs.Normal   + '%';
    document.getElementById('barPreictal').style.width = probs.Preictal + '%';
    document.getElementById('barSeizure').style.width  = probs.Seizure  + '%';
  }, 50);
  document.getElementById('pctNormal').textContent   = probs.Normal   + '%';
  document.getElementById('pctPreictal').textContent = probs.Preictal + '%';
  document.getElementById('pctSeizure').textContent  = probs.Seizure  + '%';

  // ── Alert banner ──
  const banner = document.getElementById('alertBanner');
  if (label==='Seizure') {
    banner.className='alert-banner show alert-seizure';
    document.getElementById('alertIcon').textContent='🚨';
    document.getElementById('alertText').textContent='Seizure activity detected. Immediate medical attention recommended.';
  } else if (label==='Preictal') {
    banner.className='alert-banner show alert-seizure';
    document.getElementById('alertIcon').textContent='⚠️';
    document.getElementById('alertText').textContent='Pre-seizure activity detected. Monitor the patient closely.';
  } else {
    banner.className='alert-banner show alert-normal';
    document.getElementById('alertIcon').textContent='✅';
    document.getElementById('alertText').textContent='EEG signal classified as Normal. No epileptic activity detected.';
  }

  // ── CWT scalogram ──
  if (data.cwt_image_b64) {
    const img = document.getElementById('cwtImage');
    img.src = 'data:image/png;base64,' + data.cwt_image_b64;
    const cwtCard = document.getElementById('cwtCard');
    cwtCard.classList.add('visible');
    document.getElementById('cwtChInfo').textContent =
      data.num_channels ? `Ch 1 of ${data.num_channels}` : 'Ch 1';
  }

  // ── Session metrics ──
  if (data.session_metrics) {
    const sm = data.session_metrics;
    sessionLoss.push(sm.loss);
    sessionSensitivity.push(sm.sensitivity);
    document.getElementById('lossVal').textContent        = sm.loss.toFixed(3);
    document.getElementById('sensitivityVal').textContent = sm.sensitivity.toFixed(1) + '%';
    document.getElementById('sessionMetricsCard').style.display = 'block';
    // draw sparklines after layout settles
    requestAnimationFrame(() => {
      drawSparkline('lossChart',        sessionLoss,        '#f59e0b');
      drawSparkline('sensitivityChart', sessionSensitivity, '#ef4444');
    });
  }

  updatePrecautions(label);
  addHistory(label, conf, data.num_channels);
}

/* ═══════════════════════════════════════════════════════════
   CLEAR RESULT
══════════════════════════════════════════════════════════ */
function clearResult() {
  document.getElementById('resultDisplay').className = 'result-display';
  document.getElementById('emptyState').style.display  = '';
  document.getElementById('resultContent').style.display = 'none';
  document.getElementById('alertBanner').className = 'alert-banner';
  document.getElementById('cwtCard').classList.remove('visible');
}

/* ═══════════════════════════════════════════════════════════
   HISTORY
══════════════════════════════════════════════════════════ */
function addHistory(label, conf, numChannels) {
  analysisCount++;
  const time = new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const bc   = 'badge-'+label.toLowerCase();
  const src  = currentEdfFile ? currentEdfFile.name.replace('.edf','') : 'Demo';
  const chTxt= numChannels ? `${numChannels}ch` : '—';
  document.getElementById('historyEmpty').style.display  = 'none';
  document.getElementById('historyTable').style.display  = 'table';
  const row = document.createElement('tr');
  row.innerHTML = `
    <td style="font-family:var(--mono);font-size:12px;color:var(--muted)">${String(analysisCount).padStart(3,'0')}</td>
    <td style="font-family:var(--mono);font-size:12px;">${time}</td>
    <td style="font-size:11px;color:var(--muted);max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${src}">${src}</td>
    <td style="font-family:var(--mono);font-size:11px;color:var(--muted);">${chTxt}</td>
    <td><span class="badge ${bc}">${label}</span></td>
    <td style="font-family:var(--mono);font-size:13px;color:var(--accent)">${conf}%</td>
  `;
  document.getElementById('historyBody').prepend(row);
}

/* ═══════════════════════════════════════════════════════════
   PDF REPORT
══════════════════════════════════════════════════════════ */
async function downloadReport() {
  const { jsPDF } = window.jspdf;
  const doc  = new jsPDF();
  const label      = document.getElementById('resultLabel').textContent;
  const confidence = document.getElementById('confidenceVal').textContent;
  const precautions= document.getElementById('precautionsBox')?.innerText || '';
  const lossLatest = sessionLoss.length ? sessionLoss[sessionLoss.length-1].toFixed(4) : '—';
  const sensLatest = sessionSensitivity.length ? sessionSensitivity[sessionSensitivity.length-1].toFixed(1)+'%' : '—';
  const date  = new Date().toLocaleString();
  const src   = currentEdfFile ? currentEdfFile.name : 'Demo signal';

  doc.setFont('helvetica','bold'); doc.setFontSize(18);
  doc.text('NeuroScan EEG Analysis Report', 20, 20);
  doc.setFontSize(10); doc.setFont('helvetica','normal');
  doc.text('Generated: '+date, 20, 30);
  doc.text('Source: '+src, 20, 37);
  doc.setFontSize(14);
  doc.text('Prediction:',  20, 52); doc.text(label,           70, 52);
  doc.text('Confidence:',  20, 62); doc.text(confidence,      70, 62);
  doc.text('Loss (NLL):',  20, 72); doc.text(lossLatest,      70, 72);
  doc.text('Sensitivity:', 20, 82); doc.text(sensLatest,      70, 82);
  doc.setFontSize(13);
  doc.text('Precautions & Measures:', 20, 97);
  doc.setFontSize(11);
  doc.text(doc.splitTextToSize(precautions, 170), 20, 107);
  doc.save('NeuroScan_EEG_Report.pdf');
}

/* ═══════════════════════════════════════════════════════════
   PRECAUTIONS
══════════════════════════════════════════════════════════ */
function updatePrecautions(label) {
  const box = document.getElementById('precautionsBox');
  if (label==='Normal') {
    box.innerHTML=`• Maintain proper sleep cycle<br>• Avoid stress and fatigue<br>• Balanced diet and hydration<br>• Routine health checkups`;
  } else if (label==='Preictal') {
    box.innerHTML=`• Avoid triggers (stress, flashing lights)<br>• Keep patient under observation<br>• Ensure safe surroundings<br>• Keep medication ready<br>• Inform caregivers`;
  } else {
    box.innerHTML=`• Do NOT restrain the patient<br>• Place patient on side (recovery position)<br>• Remove nearby dangerous objects<br>• Do NOT put anything in mouth<br>• Call emergency services immediately<br>• Stay with patient until recovery`;
  }
}
</script>
"""
    
    content = content.replace(js_old, js_new)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Success")

if __name__ == "__main__":
    main()
