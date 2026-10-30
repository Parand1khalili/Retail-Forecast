import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const root=path.resolve(import.meta.dirname,'..');
const out=path.join(root,'outputs/yoghurt_backtest');
const data=JSON.parse(await fs.readFile(path.join(out,'report_data.json'),'utf8'));
const wb=Workbook.create();
const summary=wb.worksheets.add('Summary');
const sheets={Brand:wb.worksheets.add('Brands'),SubBrand:wb.worksheets.add('Subbrands'),Segment:wb.worksheets.add('Segments')};
const ink='#243247',navy='#263D5B',blue='#346AA0',orange='#CC7A30',muted='#5B6674';
const selected=data.selection.selected_model;
const find=(model,level,h=0)=>data.holdout_metrics.find(r=>r.model===model&&r.level===level&&r.horizon===h);
const col=i=>{let s='';for(i++;i;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;};

function base(sh,lastRow,lastCol){
  sh.showGridLines=false;
  const range=sh.getRangeByIndexes(0,0,lastRow,lastCol);
  range.format.font={name:'Arial',size:10,color:ink};
  range.format.rowHeight=20;
  range.format.columnWidth=17;
  range.format.verticalAlignment='center';
}
function head(sh,row,n){
  const rg=sh.getRangeByIndexes(row-1,0,1,n);
  rg.format.fill=navy;rg.format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
  rg.format.horizontalAlignment='center';rg.format.wrapText=true;rg.format.rowHeight=34;
}
function title(sh,text){sh.getRange('A2').values=[[text]];sh.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:navy};}

const sheetInfo={};
for(const [level,sh] of Object.entries(sheets)){
  const raw=data.details[level];const dc=raw.columns.indexOf('Jun actual kg');
  const dims=raw.columns.slice(0,dc);
  const columns=[...dims,'Month','Actual kg','Forecast kg','Baseline kg','Error kg','Absolute error kg','Error %','Absolute error %'];
  const rows=[];
  for(const row of raw.rows){
    for(let h=0;h<3;h++){
      rows.push([...row.slice(0,dc),['Jun 2026','Jul 2026','Aug 2026'][h],row[dc+h*3],row[dc+h*3+1],row[dc+h*3+2],null,null,null,null]);
    }
  }
  const start=6,end=start+rows.length;
  base(sh,end,columns.length);title(sh,`${level==='Segment'?'Full subcategory combinations':level==='SubBrand'?'Subbrand':'Brand'} forecast comparison`);
  sh.getRange('A3').values=[[level==='Segment'?"Source: Master Mast- Aug' 25.xlsx, Jun26 and Aug 26. Volume: kg.":'Forecast origin: May 2026. Jun-Jul-Aug predicted together.']];
  sh.getRange('A3').format.font={name:'Arial',size:10,color:muted};
  sh.getRangeByIndexes(start-1,0,1,columns.length).values=[columns];head(sh,start,columns.length);
  sh.getRangeByIndexes(start,0,rows.length,columns.length).values=rows;
  const act=col(dc+1),pred=col(dc+2),err=col(dc+4),abs=col(dc+5),pct=col(dc+6),ape=col(dc+7);
  const formulas=rows.map((_,i)=>{const r=start+1+i;return [`=${pred}${r}-${act}${r}`,`=ABS(${err}${r})`,`=IF(${act}${r}=0,"n.a.",${err}${r}/${act}${r})`,`=IF(${act}${r}=0,"n.a.",${abs}${r}/${act}${r})`];});
  sh.getRange(`${err}${start+1}:${ape}${end}`).formulas=formulas;
  sh.getRange(`${act}${start+1}:${abs}${end}`).setNumberFormat('#,##0');
  sh.getRange(`${pct}${start+1}:${ape}${end}`).setNumberFormat('0.0%');
  sh.getRange(`${act}${start+1}:${ape}${end}`).format.horizontalAlignment='right';
  sh.getRange(`A${start+1}:${col(dc)}${end}`).format.horizontalAlignment='left';
  if(level==='Segment'){
    sh.getRange('A:A').format.columnWidth=24;sh.getRange('B:C').format.columnWidth=21;
    sh.getRange('D:D').format.columnWidth=16;sh.getRange('E:E').format.columnWidth=13;
    sh.getRange('F:F').format.columnWidth=20;sh.getRange('G:G').format.columnWidth=15;
  }else sh.getRangeByIndexes(0,0,end,dc).format.columnWidth=24;
  sh.tables.add(`A${start}:${col(columns.length-1)}${end}`,true,level+'Forecasts');
  sh.freezePanes.freezeRows(start);sh.freezePanes.freezeColumns(level==='Segment'?3:dc);
  sheetInfo[level]={sheet:sh.name,columns,dimensions:dc,start:start+1,end,month:col(dc),actual:act,forecast:pred,abs,error:err};
}

base(summary,49,14);summary.tabColor=navy;title(summary,'Yoghurt forecast test: Jun-Aug 2026');
summary.getRange('A4:B6').values=[['Training ends','May 2026'],['Selected method',selected],['Target','Published RA volume (kg)']];
summary.getRange('A4:A6').format.font={name:'Arial',size:10,bold:true,color:ink};
summary.getRange('B5').format.font={name:'Arial',size:10,bold:true,color:blue};
summary.getRange('A:A').format.columnWidth=42;
summary.getRange('B:D').format.columnWidth=20;
summary.getRange('E:F').format.columnWidth=18;
summary.getRange('G:G').format.columnWidth=3;
summary.getRange('H:N').format.columnWidth=13;
summary.getRange('A8:F8').values=[['Month','Actual kg','Forecast kg','Total bias %','Brand WAPE','Segment WAPE']];head(summary,8,6);
const seg=sheetInfo.Segment,brand=sheetInfo.Brand;
for(let h=1;h<=3;h++){
  const r=8+h;const month=['Jun 2026','Jul 2026','Aug 2026'][h-1];
  summary.getRange(`A${r}`).values=[[month]];
  const sum=(sh,field)=>`SUMIFS('${sh.sheet}'!${sh[field]}${sh.start}:${sh[field]}${sh.end},'${sh.sheet}'!${sh.month}${sh.start}:${sh.month}${sh.end},A${r})`;
  summary.getRange(`B${r}:F${r}`).formulas=[[
    `=${sum(seg,'actual')}`,`=${sum(seg,'forecast')}`,`=(C${r}-B${r})/B${r}`,
    `=${sum(brand,'abs')}/${sum(brand,'actual')}`,`=${sum(seg,'abs')}/${sum(seg,'actual')}`
  ]];
}
summary.getRange('A12').values=[['Three months']];
summary.getRange('B12:F12').formulas=[['=SUM(B9:B11)','=SUM(C9:C11)','=(C12-B12)/B12',`=SUM('${brand.sheet}'!${brand.abs}${brand.start}:${brand.abs}${brand.end})/SUM('${brand.sheet}'!${brand.actual}${brand.start}:${brand.actual}${brand.end})`,`=SUM('${seg.sheet}'!${seg.abs}${seg.start}:${seg.abs}${seg.end})/SUM('${seg.sheet}'!${seg.actual}${seg.start}:${seg.actual}${seg.end})`]];
summary.getRange('B9:C12').setNumberFormat('#,##0');summary.getRange('D9:F12').setNumberFormat('0.0%');
summary.getRange('A12:F12').format.font={name:'Arial',size:10,bold:true,color:ink};
summary.getRange('A15:D15').values=[['Method','Validation WAPE','Test WAPE','Selection']];head(summary,15,4);
const methods=Object.keys(data.selection.scores).sort((a,b)=>data.selection.scores[a]-data.selection.scores[b]);
summary.getRange(`A16:D${15+methods.length}`).values=methods.map(m=>[m,data.selection.scores[m],find(m,'Segment').wape,m===selected?'Selected':'Compared']);
summary.getRange(`B16:C${15+methods.length}`).setNumberFormat('0.0%');
summary.getRange(`D16:D${15+methods.length}`).format.horizontalAlignment='center';
const chosenRow=16+methods.indexOf(selected);summary.getRange(`A${chosenRow}:D${chosenRow}`).format.fill='#EAF1FA';
summary.getRange(`A${chosenRow}:D${chosenRow}`).format.font={name:'Arial',size:10,bold:true,color:navy};
summary.getRange('A27:E27').values=[['Reporting level','Actual kg','Forecast kg','WAPE','Bias']];head(summary,27,5);
for(const [i,level] of ['Total','Brand','SubBrand','Segment','City segment'].entries()){
  const r=28+i;const mm=find(selected,level);summary.getRange(`A${r}:E${r}`).values=[[level,mm.actual,mm.predicted,mm.wape,mm.bias]];
}
summary.getRange('B28:C32').setNumberFormat('#,##0');summary.getRange('D28:E32').setNumberFormat('0.0%');
summary.getRange('A35').values=[['How to read the test']];summary.getRange('A35').format.font={name:'Arial',size:12,bold:true,color:navy};
const notes=[
  'WAPE = sum of absolute errors / sum of actual sales. Lower is better.',
  'The same May origin is used for all three horizons; no June actual enters July or August forecasts.',
  'Method selection uses Sep 2025, Dec 2025 and Feb 2026 origins; all validation targets end by May 2026.',
  'Test WAPE is shown for comparison. The selected method was frozen before using Jun-Aug actuals.',
  'The two-month publication delay is not a shift in this test: May labels forecast the next three labels.',
  'Target: published retail-audit estimates. Raw calendar-month sales and shock detection are not evaluated.',
  'This first model uses historical volume and derived unit prices; no future price or distribution index.',
  'Zero actual rows retain absolute forecast errors. Percentage errors show n.a. when actual sales are zero.',
  'Brands, Subbrands and Segments are alternative views of the same market; do not add them together.',
  'Forecasts are saved model outputs; editing this workbook does not retrain the model.',
  'Baseline kg in the detail sheets repeats each group\'s May 2026 volume for all three horizons.',
];
summary.getRange(`A36:A${35+notes.length}`).values=notes.map(n=>[n]);
summary.getRange(`A36:A${35+notes.length}`).format.font={name:'Arial',size:10,color:muted};

const chart=summary.charts.add('line',summary.getRange('A8:C11'));
chart.setPosition('H8','N22');chart.title='Market volume (kg)';
chart.titleTextStyle.fontSize=12;chart.titleTextStyle.typeface='Arial';
chart.legend={position:'top',textStyle:{typeface:'Arial',fontSize:10}};
chart.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:10}};
chart.yAxis={numberFormatCode:'0.0,,"M"',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:10}};
for(const [i,s] of chart.series.items.entries())s.line={fill:i?orange:blue,style:i?'dashed':'solid',width:2};

wb.recalculate();
const checks=[];
checks.push(await wb.inspect({kind:'table',range:'Summary!A8:F12',include:'values,formulas',maxChars:3500,tableMaxRows:5,tableMaxCols:6}));
checks.push(await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:25},maxChars:2500}));
for(const c of checks)console.log(c.ndjson);
const outFile=await SpreadsheetFile.exportXlsx(wb);await outFile.save(path.join(out,'Yoghurt_Jun_Jul_Aug_2026_backtest.xlsx'));
for(const [sheetName,range,suffix] of [['Summary','A1:N33','summary'],['Brands','A1:J16','brands'],['Subbrands','A1:K16','subbrands'],['Segments','A1:K16','segments_labels'],['Segments','H6:O16','segments_values']]){
  if(process.argv.includes('--summary-only')&&sheetName!=='Summary')continue;
  const blob=await wb.render({sheetName,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(out,`preview_${suffix}.png`),new Uint8Array(await blob.arrayBuffer()));
}
console.log('EXPORTED',path.join(out,'Yoghurt_Jun_Jul_Aug_2026_backtest.xlsx'));
