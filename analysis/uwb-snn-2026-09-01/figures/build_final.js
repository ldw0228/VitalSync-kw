const fs=require('fs');
const D=require('docx');
const {Document,Packer,Paragraph,TextRun,HeadingLevel,AlignmentType,Table,TableRow,TableCell,
WidthType,ShadingType,BorderStyle,ImageRun,PageBreak,LineRuleType,Footer,PageNumber}=D;
const TR=JSON.parse(fs.readFileSync('train_v2.json','utf8'));
const W=9026, INK='1F2933', MUT='6B7680', ACC='2F6F8F', RED='B5462F', GRN='1B7A55',
      LINE='DFE3E8', BG='F4F6F7', F='Malgun Gothic';
const P=(t,o={})=>new Paragraph({spacing:{after:o.after??120,line:o.line??300,lineRule:LineRuleType.AUTO},
  alignment:o.align,
  children:(Array.isArray(t)?t:[t]).map(x=>typeof x==='string'
    ? new TextRun({text:x,size:o.size??20,color:o.color??INK,font:F,bold:o.bold}) : x)});
const R=(t,o={})=>new TextRun({text:t,size:o.size??20,color:o.color??INK,font:F,bold:o.bold});
const B=t=>R(t,{bold:true});
const H=(t,l)=>new Paragraph({heading:l,spacing:{before:l===HeadingLevel.HEADING_1?340:260,after:130},
  children:[new TextRun({text:t,font:F,color:l===HeadingLevel.HEADING_1?ACC:INK,
    size:l===HeadingLevel.HEADING_1?28:23,bold:true})]});
const H1=t=>H(t,HeadingLevel.HEADING_1), H2=t=>H(t,HeadingLevel.HEADING_2);
function cell(txt,w,o={}){
  const kids=(Array.isArray(txt)?txt:[txt]).map(x=>typeof x==='string'
    ? P(x,{after:0,size:o.size??17,bold:o.bold,color:o.color,align:o.align,line:260}) : x);
  return new TableCell({width:{size:w,type:WidthType.DXA},
    shading:o.fill?{type:ShadingType.CLEAR,fill:o.fill,color:'auto'}:undefined,
    margins:{top:65,bottom:65,left:105,right:105},children:kids});
}
function tbl(cols,head,rows,o={}){
  const bd={style:BorderStyle.SINGLE,size:4,color:LINE};
  const hr=new TableRow({tableHeader:true,children:head.map((h,i)=>
    cell(h,cols[i],{bold:true,fill:BG,size:16,align:i&&!o.leftAll?AlignmentType.CENTER:undefined}))});
  const rr=rows.map(r=>{const cs=Array.isArray(r)?r:r.c, meta=Array.isArray(r)?{}:r;
    return new TableRow({children:cs.map((c,i)=>
      cell(c,cols[i],{size:o.size??16,bold:meta.bold,color:meta.color,fill:meta.fill,
        align:i&&!o.leftAll?AlignmentType.CENTER:undefined}))});});
  return new Table({columnWidths:cols,width:{size:cols.reduce((a,b)=>a+b,0),type:WidthType.DXA},
    borders:{top:bd,bottom:bd,left:bd,right:bd,insideHorizontal:bd,insideVertical:bd},rows:[hr,...rr]});
}
const GAP=(h=140)=>new Paragraph({spacing:{after:h},children:[]});
function img(f,w,h,cap){
  const o=[new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:150,after:70,line:240,lineRule:LineRuleType.AUTO},
    children:[new ImageRun({type:'png',data:fs.readFileSync(f),transformation:{width:w,height:h}})]})];
  if(cap) o.push(P(cap,{align:AlignmentType.CENTER,size:15,color:MUT,after:200}));
  return o;
}
function note(title,lines,col){
  const c=col||ACC; const bd={style:BorderStyle.SINGLE,size:4,color:LINE};
  return new Table({columnWidths:[W],width:{size:W,type:WidthType.DXA},
    borders:{top:bd,bottom:bd,left:{style:BorderStyle.SINGLE,size:18,color:c},right:bd,
      insideHorizontal:bd,insideVertical:bd},
    rows:[new TableRow({children:[cell([P(title,{bold:true,size:18,after:55,color:c}),
      ...lines.map((l,i)=>P(l,{size:17,after:i===lines.length-1?0:55}))],W,{fill:BG})]})]});
}
const body=[]; const A=(...x)=>body.push(...x.flat());
const NM={'-':'—',rate:'rate',sf:'step-forward',pop:'population',delta:'delta',direct:'direct'};
const f2=x=>x.toFixed(2), pc=x=>Math.round(100*x)+'%';

/* ---------- 표지 ---------- */
A(P('졸업작품 · UWB 레이더 호흡 추정',{size:18,color:MUT,after:80}));
A(new Paragraph({spacing:{after:90},children:[new TextRun({
  text:'구간을 어떻게 자를 것인가, 그리고 최종 모델',font:F,size:34,bold:true,color:INK})]}));
A(P('마커 기반과 지형지물 기반의 전면 비교 · 14개 조합 학습 결과 · 작성 2026-09-01',
  {size:19,color:MUT,after:60}));
A(new Paragraph({spacing:{after:280},border:{bottom:{style:BorderStyle.SINGLE,size:6,color:LINE,space:8}},children:[]}));

/* ---------- 요약 ---------- */
A(H1('요약'));
A(P([R('레이더 신호에서 호흡수를 추정할 때, '),B('구간을 어떤 기준으로 자르는지가 어떤 모델을 쓰는지보다 결과를 크게 좌우한다'),
  R('는 것을 확인했습니다. 두 가지 자르기 방식과 여섯 가지 모델을 교차해 14개 조합을 같은 조건에서 학습했습니다.')]));
A(GAP(40));
A(tbl([2400,1500,1500,3626],
  ['','MAE (회/분)','1회/분 이내','비고'],
  [['마커 기반 · 최고','1.58','80%','SNN direct'],
   {c:['지형지물 기반 · 최고','1.18','89%','SNN delta — 최종 채택'],bold:true,fill:'EAF3EE'},
   ['고전 신호처리 (학습 없음)','1.61','79%','지형지물 기준'],
   ['ANN','1.53','86%','파라미터 17,377'],
   {c:['SNN','1.18','89%','파라미터 7,971 — 절반 이하'],bold:true}],
  {leftAll:false}));
A(GAP(160));
A(note('세 줄 결론',[
  '1. 지형지물로 자르면 어떤 방법을 써도 1.18~1.61, 마커로 자르면 1.58~1.96 — 자르기의 효과가 모델의 효과보다 큽니다.',
  '2. SNN이 ANN을 앞섭니다 (1.18 vs 1.53). 파라미터는 절반 이하입니다.',
  '3. 두 자르기 방식은 서로 5초 안에서 일치합니다 — 팀의 마커 작업이 틀렸던 것이 아니라, 두 방식이 서로를 확인해 줍니다.']));

/* ---------- 1. 왜 ---------- */
A(H1('1. 왜 이 비교가 필요했나'));
A(P([R('BIOPAC 호흡 벨트의 라벨을 레이더 신호에 옮기려면 두 장비의 시각 차를 알아야 합니다. '),
  R('실험 중 손으로 누른 '),B('마커'),R('가 그 역할을 하도록 되어 있었지만, 실제 데이터에서는 마커 개수가 '),
  B('피험자마다 5개에서 48개까지'),R(' 제각각이었습니다(기대 20~22개). ')]));
A(P('"3번째 마커가 실험2 시작"과 같은 규칙이 성립하지 않는다는 뜻입니다.'));
A(GAP(60));
A(P([B('대안 — 지형지물(地形地物) 기반. '),
  R('시각 차를 구하는 대신, 레이더가 직접 보는 사건을 찾아 그것을 기준점으로 삼습니다. '),
  R('지도 없이 산에서 위치를 잡을 때 봉우리나 강줄기를 쓰는 것과 같은 방식입니다.')]));
A(GAP(40));
A(tbl([1700,3400,3926],['실험','레이더가 보는 사건','검출 근거'],
 [['실험 1','몸을 돌릴 때 생기는 움직임 봉우리 2개','28명에서 61.0~63.3초 간격으로 일관'],
  ['실험 2','숨 참는 30초 동안의 호흡 진폭 붕괴','25명 전원 기준선의 3~13%(중앙값 7%)로 붕괴']],{leftAll:true}));
A(GAP(140));
A(note('설계상의 함정 하나',[
  '시작마커 · 회전1 · 회전2 · 종료마커가 모두 약 62초 간격이라, 아무 쌍이나 잡으면 역할이 한 칸씩 밀립니다.',
  '"회전1은 녹화 후 50~120초"라는 제약을 넣자 잘못 잡힌 4명이 모두 교정됐습니다.'],RED));

/* ---------- 2. 비교 ---------- */
A(new Paragraph({children:[new PageBreak()]}));
A(H1('2. 두 방식의 전면 비교'));
A(P([R('공정한 비교를 위해 '),B('이전 세션에서 팀과 함께 만든 HAI_동기화_종합.xlsx'),
  R('의 실험2 구간(레이더 시간)을 마커 기준으로 삼았습니다. 자르기 외의 조건 — 신호 처리, 창 길이, 추정기 — 은 모두 동일합니다.')]));
A(GAP(60));
A(H2('2.1 성능'));
A(tbl([2900,1500,1600,3026],['','MAE (회/분)','1회/분 이내','사용 가능'],
 [['마커 기반','1.78','74%','21명 · 588창'],
  {c:['지형지물 기반','1.33','82%','28명 · 784창'],bold:true,fill:'EAF3EE'}]));
A(GAP(120));
A(P([R('공통 21명 기준입니다. 21명 중 '),B('13명'),R('에서 지형지물이 낫습니다. '),
  R('구간별로 보면 평소 호흡에서 차이가 가장 크고(2.24 vs 0.65), '),
  B('회복 호흡에서는 오히려 마커가 낫습니다'),R(' — 한쪽이 전부 이기는 것은 아닙니다.')]));
A(GAP(40));
A(P([R('커버리지 차이도 큽니다. 마커 방식으로는 '),B('7명이 경계 미검출로 아예 사용 불가'),
  R('입니다. 지형지물 기반은 S01(본 녹화 12.5초)을 제외한 28명 전원을 씁니다.')]));

A(H2('2.2 두 방식은 서로를 확증한다'));
A(P('"몇 번째 마커"가 아니라 "가장 가까운 마커"로 대조하면 21명이 5초 안에서 일치합니다.'));
A(GAP(40));
A(tbl([3200,2000,1900,1926],['기준점','평균 차','표준편차','판정'],
 [['실험1 시작','+4.8초','3.3초','일치'],
  ['실험1 종료','−5.2초','3.4초','일치'],
  ['실험2 시작','+4.1초','3.4초','일치']]));
A(GAP(120));
A(P([R('계통적으로 몇 초 어긋나는 것은 '),B('정의의 차이'),
  R('입니다 — 마커는 실험을 시작하는 순간 누른 것이고, 우리 블록은 회전 봉우리를 기준으로 잡습니다. '),
  B('팀의 마커 작업이 틀렸던 것이 아닙니다.')]));

A(H2('2.3 동기 오차가 결과를 얼마나 오염시키나'));
A(P('인위적으로 시차를 넣어 재봤습니다.'));
A(GAP(40));
A(tbl([2426,1300,1300,1300,1300,1400],['넣은 시차','0초','2초','5초','10초','20초'],
 [['MAE (회/분)','1.55','1.66','1.98','2.74','4.72']]));
A(GAP(120));
A(P([R('우리 동기 정밀도는 중앙값 1.1초(σ 4.5초)입니다. '),
  B('호흡수 추정에는 충분'),R('하지만, 마커 기반이 가질 뻔했던 ±20초 오차였다면 치명적이었을 것입니다.')]));
A(GAP(140));
A(note('결정',['지형지물로 자르고, 마커로 검증하고, 안 맞는 사람을 걸러낸다.',
  '피험자별 전체 구간표와 마커와의 시차는 실험비교_마커vs지형지물.xlsx의 1_구간표 시트에 있습니다.'],GRN));

/* ---------- 3. 방법론 교정 ---------- */
A(new Paragraph({children:[new PageBreak()]}));
A(H1('3. 방법론 오류와 교정'));
A(P([R('학습 코드를 확인한 결과, 매 에폭마다 '),B('평가 폴드'),
  R('에서 성능을 재고 그중 최고를 결과로 채택하고 있었습니다. '),
  R('이는 early stopping이 아니라 '),B('평가셋으로 에폭을 고르는 것'),R('이며, 결과가 부풀려집니다.')]));
A(GAP(40));
A(tbl([3300,1900,1900,1926],['','부풀려진 값','교정 후','차이'],
 [['마커 · ANN','1.53','1.77','+0.24'],
  ['지형지물 · ANN','1.24','1.53','+0.29'],
  ['지형지물 · SNN 최고','1.13','1.18','+0.05']]));
A(GAP(120));
A(P([B('부풀림이 조건마다 다릅니다. '),
  R('그래서 "비교는 어차피 공정하니 괜찮다"고 넘길 수 없었습니다. 실제로 '),
  B('결론이 바뀌었습니다'),R(' — 이전에는 ANN과 SNN이 거의 같아 보였는데(1.24 vs 1.13), 교정 후에는 SNN이 확실히 앞섭니다.')]));
A(GAP(60));
A(H2('교정한 절차'));
A(tbl([2200,3000,3826],['','설정','왜'],
 [['분할','피험자 단위 학습/검증/평가 3분할','평가 폴드 f, 검증 폴드 (f+1)%4, 나머지 학습'],
  ['에폭','상한 60, 인내 10 조기종료','사람이 에폭을 정하지 않고 데이터가 정하게'],
  ['모델 선택','검증셋 최고 에폭의 가중치 복원','평가셋은 마지막에 한 번만 사용'],
  ['교차검증','피험자 분리 4-fold','같은 사람의 창이 학습·평가에 함께 들어가면 누수']],{leftAll:true}));

/* ---------- 4. 최종 결과 ---------- */
A(new Paragraph({children:[new PageBreak()]}));
A(H1('4. 최종 14개 조합'));
A(P('자르기 2방식 × (고전 신호처리 + ANN + SNN 인코딩 5종). 두 조건 외에는 층 구조, 폭, 학습률, 배치, 손실, 시드를 모두 고정했습니다.'));
A(GAP(60));
const best=Math.min(...TR.map(r=>r.mae));
const rows=[];
for(const [mode,lab] of [['marker','마커'],['landmark','지형지물']]){
  const sub=TR.filter(r=>r.mode===mode);
  rows.push({c:[lab,'고전 신호처리','—',f2(sub[0].base_mae),pc(sub[0].base_p1),'0','—'],fill:BG});
  for(const r of sub){
    const isBest=Math.abs(r.mae-best)<1e-9;
    rows.push({c:[lab,r.kind,NM[r.enc],f2(r.mae),pc(r.p1),r.params.toLocaleString(),
      r.stop_epochs.join(', ')],bold:isBest,fill:isBest?'EAF3EE':undefined});
  }
}
A(tbl([1150,1050,1500,1250,1200,1300,1576],
  ['자르기','모델','인코딩','MAE','1회/분','파라미터','조기종료 에폭'],rows,{size:15}));
A(GAP(160));
A(img('fig_final.png',600,393,'14개 조합 전체 순위, 자르기 효과와 모델 효과의 비교, 조기종료 에폭 분포'));

A(H2('읽어야 할 것 세 가지'));
A(P([B('1. 자르는 방식이 모델보다 중요하다. '),
  R('지형지물 안에서 모델을 바꿔 얻은 폭은 0.35인데, 자르기를 바꿔 얻은 폭은 0.48입니다. '),
  R('전처리와 라벨 품질이 모델 선택보다 큰 영향을 준다는, 흔하지만 자주 무시되는 사실의 사례입니다.')]));
A(P([B('2. SNN이 ANN을 이긴다 — 파라미터는 절반 이하로. '),
  R('1.18 vs 1.53, 7,971 vs 17,377. 조기종료 에폭을 보면 ANN은 평균 10.3에폭에서 멈추고 '),
  R('성적 상위 인코딩은 delta 19.5, rate 29.5까지 갑니다. ANN이 더 빨리 과적합합니다. '),
  R('다만 population은 평균 6.8에폭으로 더 일찍 멈추면서 성능은 ANN보다 좋습니다 — '),
  B('"오래 학습되면 좋다"는 단순 대응은 성립하지 않습니다.')]));
A(P([B('3. 인코딩 순위가 자르는 방식에 따라 뒤집힌다. '),
  R('마커 기반에서는 direct가 최고, 지형지물에서는 delta가 최고입니다. '),
  R('앞선 분류 과제(각도·호흡상태)에서는 두 과제 모두 rate가 1위로 순위가 같았는데, 이 회귀 과제에서는 유지되지 않습니다. '),
  B('인코딩 선택에 하나의 정답은 없습니다.')]));

/* ---------- 5. 파형 ---------- */
A(H1('5. 잘 맞는 파형과 안 맞는 파형'));
A(img('fig_waves.png',600,300,'위 3명은 오차 0.06~0.10 회/분, 아래 3명은 3.17~6.96 회/분'));
A(P([R('주목할 점은 '),B('못 맞힌 경우에도 파형 자체는 따라간다'),
  R('는 것입니다(상관 0.62~0.86). 무너지는 지점은 파형 복원이 아니라 '),
  B('호흡수 추출'),R('입니다 — 체동이 섞여 스펙트럼에 봉우리가 두 개 생기면 엉뚱한 쪽을 고릅니다.')]));
A(GAP(60));
A(P('별도로 시도한 end-to-end 학습(신호에서 호흡수를 직접 출력)의 결과도 같은 방향을 가리킵니다.'));
A(GAP(40));
A(tbl([3800,1500,1500,2226],['조건','창','MAE','1회/분 이내'],
 [['아무것도 학습 안 하고 평균만 답하기','784','5.87','11%'],
  ['호흡수 직접 예측','784','5.26','15%'],
  ['호흡수 직접 예측 (데이터 5배)','3,696','5.03','19%'],
  ['파형 출력 (위상 둔감 손실)','3,696','6.42','41%'],
  {c:['고전 신호처리','3,696','1.55','80%'],bold:true}],{leftAll:true}));
A(GAP(120));
A(P([R('창을 5배로 늘려도 5.26 → 5.03으로 제자리입니다 — '),
  B('표본 수가 아니라 정보량의 문제'),R('입니다. 반면 정답을 숫자 1개에서 파형 100개로 늘리자 '),
  R('1회/분 이내가 19% → 41%로 두 배가 됐습니다. '),
  B('정답 밀도가 학습을 좌우합니다.')]));
A(GAP(140));
A(note('참고 — MoRe-Fi (SenSys 2021)와의 규모 차이',[
  '논문: 12명 · 66시간 · 학습 8,000쌍 · 파라미터 2,360만 · 파형 코사인 유사도 0.9162',
  '우리: 앉아서 호흡한 구간 약 2시간 · 590~3,700창 · 파라미터 8천 · 파형 상관 0.456',
  'end-to-end가 실패한 이유가 여기 있습니다. 나머지 30명 데이터가 온 뒤에 다시 시도할 사안입니다.'],RED));

/* ---------- 6. 남은 것 ---------- */
A(H1('6. 아직 근거가 없는 설정'));
A(P('아래 세 가지는 검증셋 기준으로 탐색해야 제대로 된 설정이 됩니다. 현재는 임의로 정한 값입니다.'));
A(GAP(40));
A(tbl([2600,2200,4226],['설정','값','상태'],
 [['은닉층 크기','96 → 48, 2층','탐색 안 함. 데이터 규모를 보고 임의로 정함'],
  ['보조 특징 구성','13종','어느 것이 실제로 기여하는지 미확인'],
  ['손실 가중','회귀 : 순위 = 1 : 0.5','탐색 안 함']],{leftAll:true}));
A(GAP(140));
A(P([R('나머지 설정의 근거는 '),B('실험비교_마커vs지형지물.xlsx'),R('의 '),B('5_판단근거'),
  R(' 시트에 설정 19개 각각에 대해 "표준인지 우리가 정한 것인지"까지 적어 두었습니다.')]));
A(GAP(60));
A(H2('회의에서 확인·요청할 것'));
A(tbl([3000,6026],['항목','내용'],
 [['공통 트리거','다음 촬영 시 레이더·BIOPAC을 동시에 시작시키는 신호. 있으면 이 문서의 1~2장 전체가 불필요해집니다'],
  ['환경 파일','빈 방 1분 녹화가 전달된 30명 폴더에 없습니다. 되먹임 필터로 대체했으나 실제 참조가 더 정확합니다'],
  ['S01_CMS 재촬영','본 녹화가 12.5초(501프레임)뿐입니다'],
  ['파싱 수정본 공유','sync_tool_S02.m 190번 줄이 인터리브 방식입니다. 선배 쪽 파형 복원도 같은 영향을 받을 수 있습니다']],{leftAll:true}));

A(GAP(200));
A(new Paragraph({spacing:{before:200,after:60},border:{top:{style:BorderStyle.SINGLE,size:6,color:LINE,space:8}},children:[]}));
A(P('상세 기록은 저장소의 docs/uwb-snn-2026-09-01/ 에 있습니다. 실험 16개가 각각 폴더 하나로 정리되어 있습니다.',
  {size:16,color:MUT}));
A(P('이덕원 · 2026-09-01 · $4',{size:16,color:MUT}));

const doc=new Document({
  styles:{default:{document:{run:{font:F,size:20,color:INK},
    paragraph:{spacing:{line:300,lineRule:LineRuleType.AUTO}}}}},
  sections:[{properties:{page:{margin:{top:1100,bottom:1100,left:1180,right:1180}}},
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,
      children:[new TextRun({children:[PageNumber.CURRENT],font:F,size:16,color:MUT})]})]})},
    children:body}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync('보고서_자르기와_최종모델.docx',b);console.log('ok');});
