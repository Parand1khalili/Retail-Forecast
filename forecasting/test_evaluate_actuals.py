import tempfile, unittest, csv, json
from pathlib import Path
from zipfile import ZipFile
from xml.sax.saxutils import escape
import evaluate_actuals as m


def xlsx(path):
    pred_names=list(m.META)+['Actual | Jan-Feb 2026','Actual | Mar-Apr 2026','Actual | May-Jun 2026','Forecast | Jul-Aug 2026','Forecast | Sep-Oct 2026','Forecast | Nov-Dec 2026']
    source_names=list(m.ORIGINAL)+['Measure','Jan-Feb 2026','Mar-Apr 2026','May-Jun 2026','Jul-Aug 2026']
    def row(market,location):
        return [market,location,'SKU','A 200gr Cup','Liq','Pasteurized','[200-300)Gr','Cup','A','Co']
    def pmeta(market,location):
        z=dict(zip(m.ORIGINAL,row(market,location)))
        return [z['Market'],z['City/Region/Dis. Channel'],z['Element Caption'],z['Element Description'],z['Subcategory'],z['Subcategory 2'],z['SKUSize'],z['SKUPackage'],z['Brand'],z['Company']]
    values=[(pmeta('All','All'),[10,12,14,16,18,20],[100,120,140,155,165,175]),
            (pmeta('City','Tehran'),[5,6,7,8,9,10],[50,60,70,80,90,100])]
    def meta_values(source):
        d=dict(zip(m.ORIGINAL,source))
        return [d[{'City':'City/Region/Dis. Channel','Subcategory2':'Subcategory 2'}.get(x,x)] for x in m.META]
    v=[pred_names]+[meta_values(x)+([volume[0],volume[1],volume[2],volume[3]+1,volume[4],volume[5]] if x[0]=='All' else volume) for x,volume,_ in values]
    w=[pred_names]+[meta_values(x)+vv for x,_,vv in values]
    raw=[source_names]
    for x,volume,amount in values:
        raw.append(x+['Value Sales (1,000 Toomans)']+amount[:4]);raw.append(x+['Volume Sales(Gr)']+volume[:4])
    def col(n):
        s=''
        while n:n,r=divmod(n-1,26);s=chr(65+r)+s
        return s
    def xml(rows,blank=False):
        parts=[]
        for i,row in enumerate(rows,1):
            cells=[]
            for j,value in enumerate(row,1):
                pos=j+(1 if blank and j>10 else 0)
                if isinstance(value,(int,float)):
                    cells.append(f'<c r="{col(pos)}{i}"><v>{value}</v></c>')
                else:
                    cells.append(f'<c r="{col(pos)}{i}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
            parts.append(f'<row r="{i}">'+''.join(cells)+'</row>')
        return '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+''.join(parts)+'</sheetData></worksheet>'
    with ZipFile(path,'w') as z:
        for i,rows in enumerate([v,w,raw],1):z.writestr(f'xl/worksheets/sheet{i}.xml',xml(rows,blank=i==2))

class T(unittest.TestCase):
    def test_join_units_blank_col_market_shares_and_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.xlsx';xlsx(p)
            q=m.analyze(p,Path(d)/'out')
            self.assertEqual(q['historical_compared'],12)
            self.assertEqual(q['historical_differences'],0)
            self.assertEqual(q['matched_measure_rows'],4)
            self.assertEqual(q['unmatched_forecast'],0)
            with open(Path(d)/'out'/'scorecard.csv',encoding='utf-8-sig') as f:score=list(csv.DictReader(f))
            a=next(r for r in score if r['Metric']=='Volume' and r['Market']=='all')
            self.assertAlmostEqual(float(a['wape_pct']),100/16)
            self.assertEqual(float(a['forecast']),17)
            b=next(r for r in score if r['Metric']=='Value' and r['Market']=='city')
            self.assertAlmostEqual(float(b['forecast']),80)
            with open(Path(d)/'out'/'time_series.csv',encoding='utf-8-sig') as f:ts=list(csv.DictReader(f))
            self.assertEqual(len(ts),16)
            self.assertTrue(any(r['Period']=='Jul-Aug 2026' for r in ts))

if __name__=='__main__':unittest.main()
