/* ------------------------------------------------------------------ CONFIG */
const baseDir   = '../c_assess_inf/output/alpaca_scores';
const models    = ['gemma-2-2b-it', 'Qwen1.5-1.8B'];
const categories = [
  'voice','tone','syntax','style','special_chars',
  'obstruction','length','boundary','extra','language','context'
];

const metricNames = [
  'Task fulfilment',            'Usefulness & actionability',
  'Factual accuracy',           'Efficiency / depth',
  'Reasoning quality',          'Tone & likeability',
  'Adaptation to context',      'Safety & bias',
  'Structure & UX',             'Creativity'
];

/* ----------------------------------------------------------------- LOADING */
const modelData = {};   // { model -> { version -> [10 averages] } }

async function loadAll() {
  await Promise.all(models.map(loadModel));
  populateModelDropdown();
  updatePerModel(models[0]);
  buildAcrossModels();
}

async function loadModel(model) {
  const buckets = {};   // { paraphraseVersion -> [ [10 scores], … ] }

  await Promise.all(categories.map(async cat => {
    //const path = `${baseDir}/${model}/${cat}_json`;
    const path = `${baseDir}/${model}/${cat}.json`;
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();

      json.forEach(obj => {
        Object.entries(obj).forEach(([k,v]) => {
          if (Array.isArray(v) && v.length === 10 && typeof v[0] === 'number') {
            if (!buckets[k]) buckets[k] = [];
            buckets[k].push(v);
          }
        });
      });
    } catch (err) {
      console.warn(`⚠️ Could not load ${path}`, err);
    }
  }));

  // average per paraphrase version
  const averages = {};
  Object.entries(buckets).forEach(([version, lists]) => {
    const sums = Array(10).fill(0);
    lists.forEach(scores => scores.forEach((v,i) => sums[i] += v));
    averages[version] = sums.map(s => s / lists.length);
  });
  modelData[model] = averages;
}

/* --------------------------------------------------------- TABLE & CHARTS */
function percentile(values, p) {
  const sorted = [...values].sort((a,b) => a-b);
  const idx = (sorted.length - 1) * p;
  const lo  = Math.floor(idx), hi = Math.ceil(idx);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi]-sorted[lo])*(idx-lo);
}

function renderTable(table, data) {
  const versions = Object.keys(data).sort();
  const thead = table.querySelector('thead');
  const tbody = table.querySelector('tbody');
  thead.innerHTML = tbody.innerHTML = '';

  /* header */
  const headRow = document.createElement('tr');
  headRow.appendChild(document.createElement('th')).textContent = 'Version';
  metricNames.forEach(m => {
    const th = document.createElement('th'); th.textContent = m; headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  /* 90-percentile for highlighting */
  const perc90 = metricNames.map((_,i) => percentile(versions.map(v => data[v][i]), 0.9));

  /* body */
  versions.forEach(v => {
    const tr = document.createElement('tr');
    const tdV = document.createElement('td'); tdV.textContent = v; tr.appendChild(tdV);

    data[v].forEach((val,i) => {
      const td = document.createElement('td'); td.textContent = val.toFixed(2);
      if (val >= perc90[i]) td.classList.add('top10');
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function buildRadarChart(ctx, data, title) {
  const entries = Object.entries(data).sort((a,b) => avg(b[1]) - avg(a[1])).slice(0,5);
  return new Chart(ctx, {
    type : 'radar',
    data : {
      labels   : metricNames,
      datasets : entries.map(([v,s],i) => ({ label:v, data:s, fill:i===0 }))
    },
    options : {
      responsive:true,
      plugins:{ title:{ display:true, text:`Average metric scores – ${title}` } },
      scales  : { r:{ suggestedMin:0, suggestedMax:10, ticks:{ stepSize:2 } } }
    }
  });
}

/* -------------------------------------------------------------- PER MODEL */
let perModelChart;
function updatePerModel(model) {
  const data = modelData[model];
  const table = document.getElementById('perModelTable');
  renderTable(table, data);

  if (perModelChart) perModelChart.destroy();
  perModelChart = buildRadarChart(
    document.getElementById('perModelChart').getContext('2d'),
    data, model
  );
}

/* --------------------------------------------------------- ACROSS MODELS */
let acrossModelsChart;
function buildAcrossModels() {
  const merged = {};   // version -> [ [10 scores model-avg], … ]

  models.forEach(m => {
    Object.entries(modelData[m]).forEach(([v,s]) => {
      if (!merged[v]) merged[v] = [];
      merged[v].push(s);
    });
  });

  const averages = {};
  Object.entries(merged).forEach(([v,list]) => {
    const sums = Array(10).fill(0);
    list.forEach(scores => scores.forEach((val,i) => sums[i]+=val));
    averages[v] = sums.map(s => s / list.length);
  });

  const table = document.getElementById('acrossModelsTable');
  renderTable(table, averages);

  if (acrossModelsChart) acrossModelsChart.destroy();
  acrossModelsChart = buildRadarChart(
    document.getElementById('acrossModelsChart').getContext('2d'),
    averages, 'across models'
  );
}

/* --------------------------------------------------------------- UTIL/UX */
const avg = arr => arr.reduce((a,b)=>a+b,0) / arr.length;

function populateModelDropdown() {
  const sel = document.getElementById('modelSelect');
  models.forEach(m => {
    const opt = document.createElement('option'); opt.value = m; opt.textContent = m; sel.appendChild(opt);
  });
  sel.addEventListener('change', e => updatePerModel(e.target.value));
}

/* simple tab-switcher */
document.querySelectorAll('.tablink').forEach(btn =>
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tablink').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  })
);

/* ---------------------------------------------------------------- RUN! */
loadAll();
