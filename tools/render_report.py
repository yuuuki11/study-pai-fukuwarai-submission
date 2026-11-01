from pathlib import Path
import json, re, html
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from PIL import Image as PILImage
r=Path(__file__).resolve().parents[1]
pages=re.split(r'(?m)(?=^# )',(r/'docs/report.md').read_text())
pages=[p for p in pages if p.strip()]
pdfmetrics.registerFont(TTFont('Japanese', str(r/'tools/fonts/NotoSansJP-Regular.ttf')))
font='Japanese'
styles={
'h1':ParagraphStyle('h1',fontName=font,fontSize=22,leading=31,textColor=colors.HexColor('#142B40'),spaceAfter=18,wordWrap='CJK'),
'h2':ParagraphStyle('h2',fontName=font,fontSize=14,leading=21,spaceAfter=15,wordWrap='CJK'),
'h3':ParagraphStyle('h3',fontName=font,fontSize=12,leading=18,textColor=colors.HexColor('#14736C'),spaceBefore=13,spaceAfter=7,wordWrap='CJK'),
'p':ParagraphStyle('p',fontName=font,fontSize=10,leading=17,spaceAfter=10,wordWrap='CJK'),
't':ParagraphStyle('t',fontName=font,fontSize=9,leading=14,wordWrap='CJK')}
def para(x,style='p'):return Paragraph(html.escape(x),styles[style])
story=[]
for n,page in enumerate(pages):
 lines=page.splitlines();i=0
 while i<len(lines):
  l=lines[i]
  if not l.strip(): i+=1;continue
  if l.startswith('|'):
   rows=[]
   while i<len(lines) and lines[i].startswith('|'):
    cells=[x.strip() for x in lines[i].strip('|').split('|')]
    if not all(re.fullmatch('[-: ]+',x) for x in cells):rows.append([para(x,'t') for x in cells])
    i+=1
   count=len(rows[0]); widths=([125,370] if count==2 else [495/count]*count)
   t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E0EFEC')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.35,colors.HexColor('#CBD5DA'))]));story+=[t,Spacer(1,10)];continue
  style='h3' if l.startswith('### ') else 'h2' if l.startswith('## ') else 'h1' if l.startswith('# ') else 'p'
  story.append(para(re.sub(r'^#{1,3} ','',l),style));i+=1
 if n==0:
  story.append(Image(str(r/'docs/images/scene.jpg'),width=176,height=132))
  story.append(para('図：開始時の俯瞰画像。完成状態ではない。','t'))
 if n<len(pages)-1:story.append(PageBreak())
def footer(c,d):
 c.setStrokeColor(colors.HexColor('#CBD5DA'));c.line(50,45,545,45);c.setFont(font,8);c.setFillColor(colors.HexColor('#526271'));c.drawString(50,30,'ロボット福笑い / 最終課題');c.drawRightString(545,30,str(d.page))
doc=SimpleDocTemplate(str(r/'report.pdf'),pagesize=(595.28,841.89),rightMargin=50,leftMargin=50,topMargin=45,bottomMargin=60,title='押して作るロボット福笑い：ACT模倣学習',author='',subject='最終課題レポート')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(r/'report.pdf')
