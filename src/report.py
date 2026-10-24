"""Evidence-led five-page course report plus references. Requires real score image."""
import json,html,sys
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts';E=ROOT/'evidence';O=ROOT/'output';O.mkdir(exist_ok=True)
pdfmetrics.registerFont(TTFont('Arial','/System/Library/Fonts/Supplemental/Arial.ttf'));pdfmetrics.registerFont(TTFont('ArialB','/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialB',italic='Arial',boldItalic='ArialB')
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyX',fontName='Arial',fontSize=9,leading=12,spaceAfter=6));styles.add(ParagraphStyle(name='SmallX',fontName='Arial',fontSize=7.3,leading=9.4,spaceAfter=4));styles.add(ParagraphStyle(name='TitleX',fontName='ArialB',fontSize=21,leading=24,spaceAfter=10));styles.add(ParagraphStyle(name='HeadX',fontName='ArialB',fontSize=12,leading=15,spaceBefore=8,spaceAfter=7))
W=511;story=[]
def p(s,small=False):return Paragraph(s,styles['SmallX' if small else 'BodyX'])
def add(s,small=False):story.append(p(s,small))
def h(s):story.append(Paragraph(s,styles['HeadX']))
def table(rows,widths,small=True):
 cells=[[p(html.escape(str(v)),small) for v in row] for row in rows]
 t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf1f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#9aaeb9')),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#dce3e8')),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]));story.append(t);story.append(Spacer(1,7))
def footer(c,d):
 c.setFont('Arial',7);c.setFillColor(colors.HexColor('#586976'));c.drawString(42,25,'Khamzat Yusupov | LLM4Rec final exam | 24 October 2026');c.drawRightString(553,25,str(d.page))
def main():
 r=json.loads((A/'validation_results.json').read_text());v=json.loads((A/'verification.json').read_text());audit=json.loads((A/'data_audit.json').read_text());k=json.loads((E/'kaggle_result.json').read_text());f=json.loads((A/'final_fit.json').read_text());rec=json.loads((A/'recency_results.json').read_text()) if (A/'recency_results.json').exists() else None
 finalmethod=k.get('method','RRF: 50% personal frequency + 50% global popularity')
 story.append(Paragraph('TV programme recommendation<br/>from sparse labelled airtime',styles['TitleX']))
 add('<b>Student:</b> Khamzat Yusupov (Юсупов Хамзат) | Individual | hamzgroup@gmail.com<br/><b>Kaggle:</b> khamzatiusupov | <b>Date:</b> 2026-10-24 | GSB final exam')
 add('<b>Repository:</b> <link href="https://github.com/vhamz/sweettv-final-exam" color="#16637b">github.com/vhamz/sweettv-final-exam</link>')
 h('Abstract')
 add('This study predicts five sweet.tv programmes per user despite most broadcast rows lacking an item ID. Content profiles and implicit matrix factorization are compared with popularity and personal-frequency baselines on a shared temporal hold-out, followed by reciprocal-rank fusion. The initial fusion achieves MAP@5 0.80992 locally; the designated Kaggle submission returns private '+format(k['private_score'],'.5f')+'. The evidence favours a simple, repeat-aware strategy for this protocol, but the large local-to-hidden gap limits claims about transfer.')
 h('1. Problem and related work')
 add('The service must rank programmes airing after the viewing history, including repeat episodes or broadcasts that a user may watch again. A typical case is choosing between recurring news and a film similar to a viewer\'s previous film. The target is recorded viewing, not satisfaction; missing views are not explicit dislikes. Hu, Koren and Volinsky [1] motivate confidence-weighted implicit factorization. Metadata vectorization and TF-IDF follow official scikit-learn APIs [2]; rank fusion follows [3]. Source data and submission are from the course-designated Kaggle competition [4].')
 h('2. Data construction and time boundary')
 add('For a view [a,b) and an airing [s,s+d) on the same numeric channel, overlap = max(0, min(b,s+d)-max(a,s)). A match requires at least <b>120 seconds</b>. This absolute cut excludes very short switches relative to the measured 396-second median view interval; it does not claim a percentage watched. A qualifying view-airing match contributes one event; repeated events are summed per user/programme. Programme duration comes from the schedule, never the misspelled view-log duraton field.')
 add('Rows with tv_show_id=0 are never candidate items or relevance labels. The original models drop these rows; the later channel-context sensitivity uses such views only as past channel context. Only 115,613/1,074,235 = 10.76% of early schedule rows are labelled. There are 5,847 nonzero early programme IDs; the specification\'s 5,848 distinct values includes zero. A single ID can name several episodes; displayed titles are representative schedule strings, not a canonical episode identity.')
 add('Train: recorded wall-clock times before <b>18 May 2020 00:00</b>; validation: 18 May to <b>27 July 2020 00:00</b>. Intervals crossing a boundary are clipped. Input timestamps contain no timezone offset, so they are compared as supplied wall-clock values. Final fitting uses weeks 11-30 only. No views from weeks 31-42 are read or constructed. Future schedules provide candidate availability; content vectors use only metadata available before the fitting cutoff.')
 story.append(PageBreak())
 h('3. Methods and common evaluation')
 add('<b>Popularity:</b> number of distinct training viewers. <b>Personal frequency:</b> log(1+qualifying events). <b>Content:</b> category, three genres, production decade, directors and comma/semicolon-separated actors form sparse TF-IDF vectors; the normalized user profile is the log-count-weighted sum of watched-item vectors, ranked by cosine similarity. Vocabulary and IDF are fitted only on training items. <b>Implicit MF:</b> minimize sum c_ui(p_ui-x_u·y_i)^2 + lambda(||X||²+||Y||²), with p_ui=1 for observed events and c_ui=1+20 log(1+n_ui); absent pairs have weak confidence 1, not a recorded negative. Own NumPy ALS solves the normal equations [1].')
 add('MF trials: (32 factors, lambda=0.1) and (64, lambda=1), 12 alternating epochs, seed 20261024. The 32-factor configuration wins locally. All methods rank the same 6,363 labelled validation IDs; watched items remain eligible. Each list has five distinct IDs, descending score then ascending numeric ID; content/frequency ties use a negligible training-popularity tie-break. Cold users receive identical popularity Top-5. MF cannot learn an unobserved item vector; these candidates rank below scoreable items. Scores are ranking signals, not probabilities.')
 add('The common cohort is 1,246 template users with at least one eligible labelled validation item, including 8 without usable training history; 14 template users with no eligible truth are excluded explicitly. AP@5 = sum over ranks 1..5 of precision@rank × hit / min(5, relevant count), then macro-averaged. This conditional cohort and match rule need not reproduce Kaggle\'s hidden labelling.')
 rows=[['Method','Hold-out MAP@5','Change vs popularity']]
 for name in ['popularity','content','mf','history','fusion']:
  rows.append([{'content':'Content profiles','mf':'Implicit MF (32)','history':'Personal frequency','fusion':'Initial RRF 50/50'}.get(name,name.title()),f"{r['metrics'][name]:.5f}",f"{r['metrics'][name]-r['metrics']['popularity']:+.5f}"])
 if rec:rows.append(['Recency/context refinement',f"{rec['map5']:.5f}",f"{rec['map5']-r['metrics']['popularity']:+.5f}"])
 table(rows,[235,130,146])
 add('Initial fusion: 0.5/(60+rank_popularity)+0.5/(60+rank_personal_frequency), with one-based ranks [3]. Both required methods were evaluated individually; their fusion weights were selected as zero. Seventeen rank-fusion settings and two MF configurations are retained, including unsuccessful trials. The initial percentile-rank attempt and its outputs are retained separately; switching to RRF improved the same hold-out. This is selection evidence, not an untouched test estimate.')
 if rec:add('After the first Kaggle result, a documented sensitivity check adds 7/14/28-day event half-lives and past channel affinities multiplied by future labelled-airtime availability. Channel intervals are capped at six hours; no future viewer information enters. All 27 fusion trials use the original hold-out and are retained. Repeated validation tuning can overfit; no new independent test split is claimed.')
 add(f"<b>Measured cost/environment:</b> preparation {audit['seconds']:.2f}s; original comparison {r['total_seconds']:.2f}s; content {r['timings']['content_seconds']:.2f}s; selected MF {r['timings']['mf_seconds']:.2f}s; final original fitting {f['seconds']:.2f}s. Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, scikit-learn 1.6.1; macOS arm64, 12 logical CPUs; OMP_NUM_THREADS/OPENBLAS_NUM_THREADS=1. Timings include the stated local work only, not Kaggle processing.",True)
 # Two examples selected by training activity, without looking at recommendation hits.
 for index,ex in enumerate(r['examples']):
  story.append(PageBreak());h('4.'+str(index+1)+(' Heavy-viewer example' if index==0 else ' Light-viewer example'))
  add(f"User <b>{ex['user_id']}</b>: {int(ex['events']):,} training events across {ex['training_distinct']} IDs; {ex['validation_distinct']} labelled validation IDs. Chosen as the {'most' if index==0 else 'least'} active warm template user in the common cohort, using training event count alone.")
  add('<b>Past preferences:</b> '+'; '.join(html.escape(x['title'])+' ['+str(x['id'])+', '+str(int(x['events']))+' events]' for x in ex['training_top'][:3])+'.',True)
  lists=dict(ex['lists'])
  if rec:
   rex=next(x for x in rec['examples'] if x['user_id']==ex['user_id']);lists['improved_fusion']=rex['lists']['improved_fusion']
  rows=[['Method / rank','Programme ID','Schedule title','Category','Hit']]
  for method in ['popularity','content','mf','history','fusion']+(['improved_fusion'] if rec else []):
   for rank,x in enumerate(lists[method],1):rows.append([({'popularity':'Popularity','content':'Content','mf':'MF','history':'Frequency','fusion':'Initial RRF','improved_fusion':'Refined RRF'}[method] if rank==1 else '')+' / '+str(rank),x['id'],x['title'],x['category'],'Yes' if x['hit'] else 'No'])
  table(rows,[74,64,264,79,30])
  if index==0:
   add('The popularity list and initial fusion both retrieve five held-out items for this broad-history viewer (AP@5=1). The frequency list includes Breakfast with 1+1, absent from this user\'s labelled hold-out; familiar is therefore not automatically predictive. Content and MF recommend films, with no held-out hits in their Top-5. These examples support the measured failure, not a universal claim that either algorithm cannot recommend television.',True)
  else:
   add('One observed film, Courage Under Fire (90053098), is too narrow a profile for this viewer. Content and MF mainly return films, but none of their Top-5 appears in the labelled hold-out. Popularity retrieves Fakty at rank 2, giving AP@5=(1/2)/5=0.1; frequency puts it at rank 3 and scores 0.06667. The film recommendations are unfamiliar in this training log, but these misses do not establish useful discovery or whether the viewer enjoyed them.',True)
   add('<b>Actual labelled validation IDs:</b> '+', '.join(map(str,ex['actual_validation']))+'.',True)
 story.append(PageBreak());h('5. Final submission, platform strategy and limitations')
 add('<b>Designated method:</b> '+html.escape(finalmethod)+'. <b>Kaggle private:</b> '+format(k['private_score'],'.5f')+'; <b>public:</b> '+format(k['public_score'],'.5f')+'. The screenshot below was supplied by the student after uploading the named file. No authenticated API check or unseen competition labels were available. Frozen leaderboard position is not used.')
 image=E/'kaggle_private_score.png';from PIL import Image as PILImage
 iw,ih=PILImage.open(image).size;story.append(Image(str(image),width=W,height=W*ih/iw));story.append(Spacer(1,5))
 add('File: '+html.escape(k['submission_file'])+'; SHA-256: '+k['submission_sha256']+'.',True)
 add('I would select the repeat-aware popularity/frequency blend as the initial platform strategy under the measured protocol, rather than replace it with either tested content or MF model. It is fast, has a clear cold-start fallback and improves the original local MAP by 0.03348. A paired user bootstrap (2,000 draws, seed 20261024) gives a conditional 95% difference interval [0.02561, 0.04177]; it ignores model-selection uncertainty. The examples explain the tradeoff: recurring news can be relevant, while a film-only profile can miss a viewer\'s later habits. This is a proposed strategy based on offline evidence, not a deployed experiment or revenue claim.')
 add('<b>Limits:</b> 89.24% of early schedule rows lack IDs, so relevance is heavily censored. Long tuning intervals may be unattended viewing; a 120-second overlap is not enjoyment. Repeated IDs can merge different episodes and titles. The log has 117 duraton/time-difference mismatches; interval endpoints are authoritative. The private 0.31291 and local 0.80992 are different measurements: the hidden match rule and user/item label distribution are unknown, so no single causal explanation for the gap is established. Lack of a log entry never proves a programme was unseen elsewhere. Candidate availability and metadata coverage also change over time.')
 h('6. Verification and AI assistance')
 add('Codex generated and debugged code, inspected supplied files, searched primary references, ran comparisons, checked examples and drafted this report. Executed checks: 240 sampled views matched an independent brute-force interval oracle exactly; all 712,133 training and 565,001 validation pairs satisfied ID/overlap/time invariants; a separate dense implicit-ALS calculation agreed within 6.7e-16; AP calculations for both users matched; the CSV has 1,260 ordered users and five distinct available nonzero IDs each. No fabricated run, future-viewing feature or online outcome is included.',True)
 personal=E/'student_verification.txt'
 add('Student actions evidenced in this session: supplied the dataset and Kaggle account, uploaded the first final CSV, and returned its completed-score screenshot. '+('Personal metric explanation: '+html.escape(personal.read_text()) if personal.exists() else 'A separate personal hand calculation has not yet been supplied; the numerical cross-checks above were performed by the agent, not claimed as student work.'),True)
 add('AI tool: Codex; model identifier is recorded in the original native session. No external pretrained model or solution code was used; NumPy/SciPy/scikit-learn APIs and cited algorithms were used. The attached original, unmodified JSONL session includes earlier homework context because this exam continues the same native thread. Human interpretation of recommendations and hidden Kaggle labels remains unverified beyond the supplied evidence.',True)
 story.append(PageBreak());h('References')
 refs=[
 '[1] Y. Hu, Y. Koren and C. Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," ICDM, 2008. Author-hosted full text: https://yifanhu.net/PUB/cf.pdf . Accessed 2026-10-24.',
 '[2] scikit-learn developers, "DictVectorizer" and "TfidfTransformer," scikit-learn 1.6.1 documentation. https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.DictVectorizer.html ; https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html . Accessed 2026-10-24.',
 '[3] G. V. Cormack, C. L. A. Clarke and S. Buettcher, "Reciprocal rank fusion outperforms condorcet and individual rank learning methods," SIGIR, pp. 758-759, 2009. https://doi.org/10.1145/1571941.1572114 . Author-hosted paper: https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf . Accessed 2026-10-24.',
 '[4] ogurbych, "sweet.tv - TV program recommender," Kaggle competition 24201. https://www.kaggle.com/competitions/sweettv-tv-program-recommender . Dataset supplied by the student; account khamzatiusupov; score evidence supplied 2026-10-24.',
 '[5] S. Jin, "LLM4Rec Homework Guidelines" and "Midterm Quiz," course materials supplied by the student, HSE University, 2026. Original local files: homework_guidelines.pdf and LLM4Rec. Midterm Quiz.pptx.']
 for text in refs:add(html.escape(text))
 add('Reproducibility: src/prepare.py, src/model.py, src/recency.py and src/verify.py; configuration and all measured local trials in artifacts/. The repository contains the exact submitted CSV and score evidence, but not redistributed raw competition data. Re-running after installing requirements.txt needs the five original files in SWEETTV_DATA.')
 doc=SimpleDocTemplate(str(O/'yusupov_khamzat_final_report.pdf'),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=40,title='TV Programme Recommendation - sweet.tv',author='Khamzat Yusupov');doc.build(story,onFirstPage=footer,onLaterPages=footer)
 print(O/'yusupov_khamzat_final_report.pdf')
if __name__=='__main__':main()
