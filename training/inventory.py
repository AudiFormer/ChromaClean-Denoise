# Scan photo folders for RAW files and record camera, ISO, size -> inventory.csv
import os, sys, csv, exifread
from concurrent.futures import ThreadPoolExecutor
EXT=('.arw','.orf','.dng','.nef','.cr2','.cr3','.raf','.rw2')
roots=sys.argv[1:]
paths=[os.path.join(dp,f) for r in roots for dp,_,fs in os.walk(r) for f in fs if f.lower().endswith(EXT) and '\denoise' not in dp and '/denoise' not in dp]
def info(p):
    try:
        with open(p,'rb') as fh: t=exifread.process_file(fh,details=False,stop_tag='EXIF ISOSpeedRatings')
        g=lambda k: str(t.get(k,'')).strip()
        return [p, g('Image Make'), g('Image Model'), g('EXIF ISOSpeedRatings'), g('EXIF ExposureTime'), os.path.getsize(p)]
    except Exception as e:
        return [p,'ERR',str(e)[:40],'','',0]
with ThreadPoolExecutor(16) as ex: rows=list(ex.map(info,paths))
with open(os.path.join(os.path.dirname(__file__),'inventory.csv'),'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['path','make','model','iso','exposure','bytes']); w.writerows(rows)
from collections import Counter
c=Counter((r[1],r[2]) for r in rows); print(len(rows),'files')
for k,v in c.most_common(): 
    isos=[int(r[3]) for r in rows if (r[1],r[2])==k and r[3].isdigit()]
    low=sum(1 for i in isos if i<=200)
    print(f'{v:6} {k[0]} {k[1]}  ISO<=200: {low}')
