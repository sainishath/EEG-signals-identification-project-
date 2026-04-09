import re
import sys

def main():
    file_path = r"d:\desktop\project file\New folder\withlogin.html"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. CSS changes
    css_old = "  --seizure: #ef4444;"
    css_new = "  --seizure: #ef4444;\n  --postictal: #8b5cf6;"
    content = content.replace(css_old, css_new)

    css_old2 = ".btn-seizure .btn-dot  { background: var(--seizure);  box-shadow: 0 0 6px var(--seizure); }"
    css_new2 = css_old2 + "\n.btn-postictal .btn-dot { background: var(--postictal); box-shadow: 0 0 6px var(--postictal); }"
    content = content.replace(css_old2, css_new2)

    css_old3 = ".result-display.result-seizure  { border-color: rgba(239,68,68,.4);  background: rgba(239,68,68,.06); }"
    css_new3 = css_old3 + "\n.result-display.result-postictal { border-color: rgba(139,92,246,.4); background: rgba(139,92,246,.06); }"
    content = content.replace(css_old3, css_new3)

    css_old4 = ".result-seizure  .result-icon { color: var(--seizure);  background: rgba(239,68,68,.1); }"
    css_new4 = css_old4 + "\n.result-postictal .result-icon { color: var(--postictal); background: rgba(139,92,246,.1); }"
    content = content.replace(css_old4, css_new4)

    css_old5 = ".result-seizure  .result-label { color: var(--seizure); }"
    css_new5 = css_old5 + "\n.result-postictal .result-label { color: var(--postictal); }"
    content = content.replace(css_old5, css_new5)

    css_old6 = ".prob-seizure  .prob-bar-fill { background: var(--seizure); }"
    css_new6 = css_old6 + "\n.prob-postictal .prob-bar-fill { background: var(--postictal); }"
    content = content.replace(css_old6, css_new6)

    css_old7 = ".badge-seizure  { background: rgba(239,68,68,.15);  color: var(--seizure); }"
    css_new7 = css_old7 + "\n.badge-postictal { background: rgba(139,92,246,.15); color: var(--postictal); }"
    content = content.replace(css_old7, css_new7)

    css_old8 = ".alert-seizure { border-color: rgba(239,68,68,.5); background: rgba(239,68,68,.1); color: #fca5a5; }"
    css_new8 = css_old8 + "\n.alert-postictal { border-color: rgba(139,92,246,.5); background: rgba(139,92,246,.1); color: #c4b5fd; }"
    content = content.replace(css_old8, css_new8)

    # 2. HTML demo buttons
    btn_old = """          <button class="demo-btn btn-seizure" onclick="loadDemo('seizure', event)">
            <div><div class="btn-label">Seizure EEG</div><div class="btn-desc">Ictal activity</div></div>
            <span class="btn-dot"></span>
          </button>"""
    btn_new = btn_old + """
          <button class="demo-btn btn-postictal" onclick="loadDemo('postictal', event)">
            <div><div class="btn-label">Postictal EEG</div><div class="btn-desc">Post-seizure recovery</div></div>
            <span class="btn-dot"></span>
          </button>"""
    content = content.replace(btn_old, btn_new)

    # 3. HTML probabilities array
    prob_old = """            <div class="prob-row prob-seizure">
              <span class="prob-name">Seizure</span>
              <div class="prob-bar-bg"><div class="prob-bar-fill" id="barSeizure"></div></div>
              <span class="prob-pct" id="pctSeizure">—</span>
            </div>"""
    prob_new = prob_old + """
            <div class="prob-row prob-postictal">
              <span class="prob-name">Postictal</span>
              <div class="prob-bar-bg"><div class="prob-bar-fill" id="barPostictal"></div></div>
              <span class="prob-pct" id="pctPostictal">—</span>
            </div>"""
    content = content.replace(prob_old, prob_new)

    # 4. JS LoadDemo colors
    load_old = "const colors = { normal:'#22c55e', preictal:'#f59e0b', seizure:'#ef4444' };"
    load_new = "const colors = { normal:'#22c55e', preictal:'#f59e0b', seizure:'#ef4444', postictal:'#8b5cf6' };"
    content = content.replace(load_old, load_new)

    # 5. analyzeSignal color drawing logic
    analyze_old = """      const col = data.predicted_label==='Seizure'?'#ef4444'
                : data.predicted_label==='Preictal'?'#f59e0b':'#22c55e';"""
    analyze_new = """      const col = data.predicted_label==='Seizure'?'#ef4444'
                : data.predicted_label==='Preictal'?'#f59e0b'
                : data.predicted_label==='Postictal'?'#8b5cf6':'#22c55e';"""
    content = content.replace(analyze_old, analyze_new)

    # 6. JS details in showResult
    show_res_old = "const icons = { Normal:'✅', Preictal:'⚠️', Seizure:'🚨' };"
    show_res_new = "const icons = { Normal:'✅', Preictal:'⚠️', Seizure:'🚨', Postictal:'🧠' };"
    content = content.replace(show_res_old, show_res_new)

    sub_old = """    Normal:   'No epileptic activity detected',
    Preictal: 'Pre-seizure state — monitor closely',
    Seizure:  'Active seizure detected — alert!'"""
    sub_new = sub_old + ",\n    Postictal: 'Post-seizure recovery state detected'"
    content = content.replace(sub_old, sub_new)

    bars_old = """    document.getElementById('barNormal').style.width   = probs.Normal   + '%';
    document.getElementById('barPreictal').style.width = probs.Preictal + '%';
    document.getElementById('barSeizure').style.width  = probs.Seizure  + '%';"""
    bars_new = bars_old + "\n    document.getElementById('barPostictal').style.width  = probs.Postictal  + '%';"
    content = content.replace(bars_old, bars_new)

    pcts_old = """  document.getElementById('pctNormal').textContent   = probs.Normal   + '%';
  document.getElementById('pctPreictal').textContent = probs.Preictal + '%';
  document.getElementById('pctSeizure').textContent  = probs.Seizure  + '%';"""
    pcts_new = pcts_old + "\n  document.getElementById('pctPostictal').textContent  = probs.Postictal  + '%';"
    content = content.replace(pcts_old, pcts_new)

    # 7. JS target Alert Banner
    alert_old = "banner.className='alert-banner show alert-normal';"
    alert_new = "banner.className='alert-banner show alert-normal';"
    alert_post = """  } else if (label==='Postictal') {
    banner.className='alert-banner show alert-postictal';
    document.getElementById('alertIcon').textContent='🧠';
    document.getElementById('alertText').textContent='Patient is in postictal phase. Recovery and rest required.';"""
    
    # insert before "else { \n    banner.className='alert-banner show alert-normal';"
    content = content.replace("  } else {\n    banner.className='alert-banner show alert-normal';", alert_post + "\n  } else {\n    banner.className='alert-banner show alert-normal';")

    # 8. JS target precautions
    prec_old = "box.innerHTML=`• Do NOT restrain the patient<br>• Place patient on side (recovery position)<br>• Remove nearby dangerous objects<br>• Do NOT put anything in mouth<br>• Call emergency services immediately<br>• Stay with patient until recovery`;"
    prec_new = """box.innerHTML=`• Do NOT restrain the patient<br>• Place patient on side (recovery position)<br>• Remove nearby dangerous objects<br>• Do NOT put anything in mouth<br>• Call emergency services immediately<br>• Stay with patient until recovery`;
  } else if (label==='Postictal') {
    box.innerHTML=`• Allow the patient to rest quietly<br>• Confusion and exhaustion are normal<br>• Do not offer food or drink until fully alert<br>• Stay with them until fully conscious<br>• Comfort and reassure the patient`;"""
    content = content.replace(prec_old, prec_new)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("HTML Update applied!")

if __name__ == "__main__":
    main()
