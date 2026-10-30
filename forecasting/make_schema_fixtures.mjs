import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const out=path.resolve(import.meta.dirname,'../outputs/schema_tests');await fs.mkdir(out,{recursive:true});
for(const manual of [false,true]){
 const wb=Workbook.create(),sh=wb.worksheets.add('Milk sales');
 sh.getRange('A1').values=[['Synthetic milk data for parser checks']];
 sh.getRange('A3:J7').values=[
  manual?['Axis','Label X','Place X','Owner X','Family X','Product X','Grade X','2026-05','Jun 2026',"Jul'26"]:['Sales Metric','Brand Name','City Name','Producer','Sub Brand','Sub Category 1','Sub Category 2','2026-05','Jun 2026',"Jul'26"],
  [manual?'Units moved':'Sales Volume (Liters)','Brand A','Tehran','Maker A','Plain','Fresh','LF',100,110,120],
  [manual?'Gross turnover':'Sales Value (Tomans)','Brand A','Tehran','Maker A','Plain','Fresh','LF',1000,1210,1440],
  [manual?'Units moved':'Sales Volume (Liters)','Brand A','Karaj','Maker A','Plain','Fresh','LF',40,50,60],
  [manual?'Gross turnover':'Sales Value (Tomans)','Brand A','Karaj','Maker A','Plain','Fresh','LF',400,550,720],
 ];
 sh.getRange('A3:J3').format.fill='#174ea6';sh.getRange('A3:J3').format.font.color='#FFFFFF';
 sh.getRange('A:J').format.columnWidth=21;
 const file=await SpreadsheetFile.exportXlsx(wb);await file.save(path.join(out,manual?'manual_headers.xlsx':'similar_headers.xlsx'));
 if(!manual){const png=await wb.render({sheetName:sh.name,range:'A1:J7',scale:1.4,format:'png'});await fs.writeFile(path.join(out,'fixture_preview.png'),new Uint8Array(await png.arrayBuffer()));}
}
console.log('Created two synthetic schema fixtures.');
