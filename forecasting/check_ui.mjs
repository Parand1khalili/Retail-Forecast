import fs from 'node:fs';
import vm from 'node:vm';
const html=fs.readFileSync(new URL('./app_ui.html',import.meta.url),'utf8');
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
if(scripts.length!==1)throw Error('Expected one inline UI script.');
new vm.Script(scripts[0][1]);
if(html.includes('id="cutoff"></select></div><div class="field"'))throw Error('Manual cutoff remains in normal user flow.');
console.log('UI JavaScript syntax and automatic flow checks passed.');
