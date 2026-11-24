// No browser dependencies: exercise the actual Model Lab renderer with fake DOM nodes.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, 'app_ui.html'), 'utf8');
const start = html.indexOf('function labMetricName(metric){');
const end = html.indexOf('function renderFuture(r){', start);
assert.ok(start >= 0 && end > start, 'Model Lab must be wired into the actual UI');
assert.ok(html.includes('id="runBacktest"'), 'Model Lab action button exists');
assert.ok(html.includes('id="downloadBacktest"'), 'Report download button exists');

function runWithData(payload) {
  const nodes = Object.fromEntries(
    ['results', 'labMetricView', 'labHorizonView', 'labLevelView',
      'labGroupCount', 'labGroupTable'].map(key => [key, {innerHTML:'', value:'', textContent:''}])
  );
  const sandbox = {
    latestLabData: null,
    el: id => {
      assert.ok(nodes[id], 'Missing Model Lab element ' + id);
      return nodes[id];
    },
    persian: n => String(n),
    percent: n => (Number(n) * 100).toFixed(1) + '%',
    escapeHtml: str => String(str).replace(/[&<>"']/g, ch => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    })[ch]),
    choice: (node,values,selected) => {
      node.value = values.some(([value]) => value === selected)
        ? selected : (values[0] ? values[0][0] : '');
    }
  };
  vm.createContext(sandbox);
  vm.runInContext(html.slice(start, end), sandbox);
  sandbox.renderBacktest(payload);
  return {nodes, sandbox};
}

const source = '<img src=x onerror=alert(1)>';
const fake = {
  source, market: 'All', periodicity: 'bimonthly', periods:36, series:16,
  preview_limit:15, metrics:['volume_kg'],
  summaries:[{
    metric:'volume_kg', horizon:'1', audit_type:'untouched_holdout',
    origin:'2026-05', legacy_global_wape:.15, global_wape:.12,
    hierarchical_wape:.08, relative_reduction_vs_legacy:.47
  },{
    metric:'volume_kg', horizon:'2', audit_type:'untouched_holdout',
    origin:'2026-05', legacy_global_wape:.17, global_wape:.16,
    hierarchical_wape:.11, relative_reduction_vs_legacy:.35
  }],
  group_totals:[
    {metric:'volume_kg',horizon:1,level:'Subcategory',
     groups:3,improved:2,deteriorated:1,unchanged:0},
    {metric:'volume_kg',horizon:2,level:'SKU',
     groups:2,improved:1,deteriorated:1,unchanged:0}
  ],
  group_preview:[
    {metric:'volume_kg',horizon:'1',level:'Subcategory',
     group:'Beverages <script>alert(1)</script>',series:'8',actual:'500',
     legacy_global_wape:'.15',global_wape:'.12',hierarchical_wape:'.08',
     relative_reduction_vs_legacy:'.47'},
    {metric:'volume_kg',horizon:'2',level:'SKU',group:'SKU-12',series:'1',
     actual:'0',legacy_global_wape:'',global_wape:'',
     hierarchical_wape:'',relative_reduction_vs_legacy:''}
  ]
};
const {nodes, sandbox} = runWithData(fake);
assert.match(nodes.results.innerHTML,/مقایسه تاریخی Recipeها/);
assert.match(nodes.results.innerHTML,/47.0%/);
assert.ok(!nodes.results.innerHTML.includes(source), 'Source HTML is escaped');
assert.match(nodes.labGroupTable.innerHTML,/Beverages &lt;script&gt;/);
assert.ok(!nodes.labGroupTable.innerHTML.includes('<script>'));
assert.match(nodes.labGroupCount.textContent,/3 گروه/);
nodes.labHorizonView.value = '2';
sandbox.labGroupLevels();
assert.match(nodes.labGroupTable.innerHTML,/SKU-12/);
assert.match(nodes.labGroupTable.innerHTML,/N\/A/);

const noHoldout = {...fake, summaries:[
  {metric:'volume_kg',horizon:'1',audit_type:'no_independent_holdout',
   origin:'',legacy_global_wape:null,global_wape:null,hierarchical_wape:null}
],group_totals:[],group_preview:[]};
const result = runWithData(noHoldout);
assert.match(result.nodes.results.innerHTML,/Holdout کافی نیست/);
assert.match(result.nodes.labGroupTable.innerHTML,/دادهٔ Holdout موجود نیست/);
// The locked Recipe comparison is not a replay of the live price override.
const guardAudit=runWithData({...fake,metrics:['value_1000_toman'],summaries:[{
  metric:'value_1000_toman',horizon:'1',audit_type:'untouched_holdout',
  origin:'2026-05',legacy_global_wape:.5,global_wape:.5,hierarchical_wape:.5,
  relative_reduction_vs_legacy:0,price_guard_active_at_holdout:true,price_guard_wape:0,
  live_audit_scope:'cutoff_safe_policy_replay',previous_live_wape:.8,live_global_wape:.6,
  live_hierarchical_wape:.4,live_relative_reduction_vs_previous:.5,live_shock_events:2,live_specialist_series:8
}],group_totals:[],group_preview:[]});
assert.match(guardAudit.nodes.results.innerHTML,/قاعده مستقل جهش ارزش واحد در مبدأ آزمون فعال بود/);
assert.match(guardAudit.nodes.results.innerHTML,/H1: 0.0%/);
assert.match(guardAudit.nodes.results.innerHTML,/بازاجرای کامل برنامه/);
assert.match(guardAudit.nodes.results.innerHTML,/بازاجرای سیاست واقعی فروش ریالی در مبدأ Holdout/);
assert.match(guardAudit.nodes.results.innerHTML,/80.0%/);
assert.match(guardAudit.nodes.results.innerHTML,/40.0%/);
assert.doesNotMatch(nodes.results.innerHTML,/قاعده مستقل جهش ارزش واحد در مبدأ آزمون فعال بود/);
// Check actual preflight renderer against ready, missing deps, stale backend and
// unavailable /api/health; never prescribe pip for a stale server.
const runtimeStart = html.indexOf('function showRuntimeStatus(health,requestError=');
const runtimeEnd = html.indexOf('function updateWizard(){',runtimeStart);
assert.ok(runtimeStart >= 0 && runtimeEnd > runtimeStart);
const runtimeNodes = Object.fromEntries(
  ['runtimeAlert','runtimeTitle','runtimeProblem','runtimeCommand','runtimeHint']
    .map(key => [key,{hidden:true,textContent:''}])
);
let wizardChecks = 0, runtimeError = '';
const runtimeSandbox = {
  engineReady:false,
  el:id=>{ assert.ok(runtimeNodes[id], 'Unknown runtime node '+id);return runtimeNodes[id]; },
  updateWizard:()=>{wizardChecks++;},
  status:msg=>{runtimeError=msg;},
};
vm.createContext(runtimeSandbox);
vm.runInContext(html.slice(runtimeStart,runtimeEnd),runtimeSandbox);

runtimeSandbox.showRuntimeStatus({service:'ra-forecast',ready:false,runtime:{dependencies:{
  ready:false,
  python_executable:'C:\\My Project\\.venv\\Scripts\\python.exe',
  failures:[{package:'catboost',reason:"No module named 'catboost'"}],
  install_command:'python -m pip install -r requirements.txt'
}}});
assert.equal(runtimeNodes.runtimeAlert.hidden,false);
assert.match(runtimeNodes.runtimeTitle.textContent,/پکیج‌های موتور/);
assert.match(runtimeNodes.runtimeProblem.textContent,/catboost/);
assert.match(runtimeNodes.runtimeProblem.textContent,/My Project/);
assert.match(runtimeNodes.runtimeCommand.textContent,/pip install -r/);
assert.match(runtimeError,/پکیج‌های موتور/);

runtimeSandbox.showRuntimeStatus({service:'ra-forecast',ready:true,runtime:{
  root:'H:\\project',python:'C:\\old-python\\python.exe',
  paths:{'app.py':'H:\\old-project\\forecasting\\app.py'}
}});
assert.equal(runtimeNodes.runtimeAlert.hidden,false);
assert.match(runtimeNodes.runtimeTitle.textContent,/نسخه سرور/);
assert.match(runtimeNodes.runtimeProblem.textContent,/old-project/);
assert.match(runtimeNodes.runtimeCommand.textContent,/Get-NetTCPConnection/);
assert.ok(!runtimeNodes.runtimeCommand.textContent.includes('pip install'),
          'A stale backend must not be misdiagnosed as missing CatBoost');
assert.match(runtimeNodes.runtimeHint.textContent,/Ctrl\+C/);

runtimeSandbox.showRuntimeStatus(null,'HTTP 404');
assert.match(runtimeNodes.runtimeTitle.textContent,/اتصال به API/);
assert.match(runtimeNodes.runtimeProblem.textContent,/HTTP 404/);
assert.ok(!runtimeNodes.runtimeCommand.textContent.includes('pip install'));

runtimeSandbox.showRuntimeStatus({service:'ra-forecast',ready:true,runtime:{
  dependencies:{ready:true,python_executable:'C:\\Python314\\python.exe'}
}});
assert.equal(runtimeSandbox.engineReady,true);
assert.equal(runtimeNodes.runtimeAlert.hidden,true);
assert.equal(wizardChecks,4);
console.log('Runtime dependency, stale-server, failed-health and ready-state UI: PASS');

console.log('Model Lab UI rendering, filters, zero-demand N/A and escaping: PASS');
