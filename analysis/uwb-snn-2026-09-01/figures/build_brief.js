const fs=require('fs');
const D=require('docx');
const {Document,Packer,Paragraph,TextRun,HeadingLevel,AlignmentType,Table,TableRow,TableCell,
WidthType,ShadingType,BorderStyle,ImageRun,PageBreak,LineRuleType,Footer,PageNumber}=D;
const ROWS=JSON.parse(fs.readFileSync('rows.json','utf8'));
const W=9026, INK='1F2933', MUT='6B7680', ACC='2F6F8F', LINE='DFE3E8', BG='F4F6F7', F='Malgun Gothic';
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
  const rr=rows.map(r=>new TableRow({children:r.map((c,i)=>
    cell(c,cols[i],{size:o.size??16,align:i&&!o.leftAll?AlignmentType.CENTER:undefined}))}));
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
function note(title,lines){
  const bd={style:BorderStyle.SINGLE,size:4,color:LINE};
  return new Table({columnWidths:[W],width:{size:W,type:WidthType.DXA},
    borders:{top:bd,bottom:bd,left:{style:BorderStyle.SINGLE,size:18,color:ACC},right:bd,
      insideHorizontal:bd,insideVertical:bd},
    rows:[new TableRow({children:[cell([P(title,{bold:true,size:18,after:55,color:ACC}),
      ...lines.map((l,i)=>P(l,{size:17,after:i===lines.length-1?0:55}))],W,{fill:BG})]})]});
}
const body=[]; const A=(...x)=>body.push(...x.flat());

A(P('졸업작품 · 회의 자료',{size:18,color:MUT,after:80}));
A(new Paragraph({spacing:{after:90},children:[new TextRun({text:'IR-UWB 원시 데이터 파싱 정정과 호흡 복원 일치도',
  font:F,size:34,bold:true,color:INK})]}));
A(P('2026-08-31 피드백에 대한 확인 결과 · 작성 2026-08-31',{size:19,color:MUT,after:60,
  }));
A(new Paragraph({spacing:{after:280},border:{bottom:{style:BorderStyle.SINGLE,size:6,color:LINE,space:8}},children:[]}));

A(H1('요약'));
A(P([R('피드백에서 지적된 '),B('"위아래로 뜨는 한 쌍(라인오브사이트 / 고스트)"'),
  R('을 실제 데이터로 확인했습니다. 결론은 두 가지입니다.')]));
A(GAP(40));
A(tbl([700,8326],['','내용'],[
  ['1','그 한 쌍은 멀티패스가 아니라 프레임 구조입니다. 간격이 정확히 92 bin(블록 크기)이고, 87건 전부에서 같은 거리에 나타납니다. 멀티패스라면 더 먼 거리에, 사람·방마다 다르게 나타나야 합니다.'],
  ['2','다만 "둘 다 쓰라"는 조언의 실질은 맞습니다. 두 줄은 상관 −0.15로 서로 다른 정보(I와 Q)를 담고 있고, 합쳐야 호흡이 복원됩니다. 그리고 파싱을 고친 뒤에도 진짜 2차 반사가 남습니다 — 레이더 3에서 +30 bin, 세기 0.43으로 일관.'],
],{leftAll:true}));
A(GAP(140));
A(P([R('그 위에서 호흡 복원 성능을 정량화하고 '),B('기준선을 확정했습니다'),
  R('. 실험 1·2만으로 정박하면 29명 중 '),B('28명'),R('에서 동기화가 되고(이전 25명), 레이더 3대의 호흡수 추정을 중앙값으로 합치면 '),
  B('MAE 1.08 BPM · 28명 중 20명이 1 BPM 이내, 25명이 2.5 BPM 이내'),R('입니다. 학습은 아직 하나도 들어가지 않았습니다.')]));
A(P([R('그리고 '),B('매 순간 가장 좋은 레이더를 고를 수 있다면 MAE 0.32 BPM'),
  R('까지 내려갑니다. 이 1.08 → 0.32 구간이 모델이 파고들 자리이고, 모델의 임무는 "호흡을 읽는 것"이 아니라 '),
  B('시점마다 어느 레이더를 믿을지 고르는 것'),R('으로 좁혀집니다.')]));
A(GAP(80));
A(note('요청하신 BIOPAC 점검 결과',[
  '평소 호흡 구간에서 25 BPM을 넘는 피험자는 0명입니다. 호흡 주기 변동계수도 최대 0.27로 전원 규칙적입니다.',
  '중앙값 14.5 BPM, 범위 9.7~20.3. 가장 낮은 S03_PSJ·S04_KTW(9.7)는 변동계수가 0.18/0.08로 매우 규칙적이라 센서 문제가 아니라 느리고 안정적인 호흡으로 보입니다.',
  '앞서 S22_KJH의 BIOPAC이 이상하다고 적었으나, 추정 방식을 개선하니 BIOPAC 쪽은 정상이었습니다(느린 호흡 6.1 BPM). 이 피험자의 문제는 레이더 쪽 선택 실패입니다 — 2장 참조.']));

A(GAP(120));
A(note('환경 파일(빈 방 녹화)은 전달받은 데이터에 없습니다',[
  '피험자 30명 폴더 전부가 레이더 3대 × 녹화 1개씩입니다. 빈 방 녹화가 들어 있는 폴더는 없습니다.',
  '유일한 예외인 S01_CMS의 추가 녹화(24분, "S48_1S67_MS")를 열어봤으나 사람이 있습니다 — bin 30에서 11.7 BPM 호흡이 또렷합니다(피크/중앙값 비 21.6). 파일럿 촬영으로 보입니다.',
  '주신 sync_tool_S02.m 에도 환경 파일을 불러오는 부분은 없습니다.',
  '나머지 30명 데이터를 보내주실 때 환경 파일도 함께 부탁드립니다.']));

A(new Paragraph({children:[new PageBreak()]}));
A(H1('1.  "고스트"의 정체'));
A(P('레이더별 가슴 위치를 재던 중, 모든 레이더에서 호흡 대역 봉우리가 정확히 46 bin 간격으로 두 개씩 나타나는 것을 발견했습니다. 46은 92의 절반입니다. 실제 파일로 세 가지 파싱을 비교한 결과, 한 프레임 185개 값은 [프레임 카운터][실수 92][허수 92] 구조였습니다. 기존 코드는 짝수·홀수 위치에서 번갈아 뽑아 복소수를 만들고 있었고, 그 결과 위상이 무의미한 값이 되어 크기만 쓸 수 있었습니다.'));
A(img('fig_ghost.png',600,383,
  'A: 184개 값을 그대로 거리축으로 보면 한 쌍이 뜹니다 — 간격이 정확히 92. B: 올바르게 나누면 한 줄로 모이고, 그 위에 진짜 2차 반사가 남습니다. C: r1·r3이 등거리, r2만 멉니다.'));
A(GAP(60));
A(tbl([1500,2900,2400,2226],['파싱 방식','복소 복원 규칙','호흡대역 피크','판정'],[
  ['A · 인터리브 (기존)','d(:,1:2:end) + 1i·d(:,2:2:end)','bin 14~16 그리고 59~60','같은 표적이 46 bin 떨어져 중복'],
  ['B · 전후분할 (채택)','d(:,1:92) + 1i·d(:,93:end)','bin 28~32 — 봉우리 하나','물리적으로 타당'],
  ['C · 복소 아님','184개를 모두 실수 bin으로','bin 28~32 그리고 123~127','앞·뒤 블록에 같은 표적 → 구조 증명'],
],{leftAll:true}));
A(GAP(120));
A(P([B('수정 후 검증. '),R('레이더에서 뽑은 호흡 주파수가 BIOPAC과 일치했습니다 — S05 0.186 vs 0.176 Hz · S03 0.107 vs 0.117 Hz · S09 0.244 vs 0.244 Hz.')]));

A(H2('두 줄은 복사본이 아닙니다'));
A(P('"거의 비슷해 보이지만 약간 다른 파형"이라는 지적은 정확합니다. 두 줄의 파형 상관은 −0.15로, 실제로 다른 신호입니다. 그런데 다르다는 것이 표적이 둘이라는 뜻은 아닙니다 — 같은 표적의 위상을 I축과 Q축에 투영한 값이라 이렇게 됩니다.'));
A(img('fig_iq.png',600,238,
  '왼쪽: 두 줄의 파형은 실제로 다릅니다. 오른쪽: 그러나 거리 차이로 보면 87건 중 97%가 ±3 bin 안 — 같은 거리입니다. 진짜 2차 반사는 −25~+33으로 제각각입니다.'));
A(GAP(60));
A(note('그래서 조언의 실질은 유효합니다',[
  '"아래 것만 쓰지 말고 위 것도 같이 써라" — 두 줄이 서로 다른 정보를 담고 있으므로 맞는 말입니다.',
  '현재 파이프라인은 이미 실수·허수를 복소수로 합쳐 I/Q 평면 주성분에 투영하고, 크기까지 함께 사용합니다.',
  '차이는 "위쪽을 별도 표적으로 두 번 계산하느냐" vs "하나의 복소 신호로 합치느냐"이고, 후자가 위상을 살리는 방식입니다.',
  '멀티패스 활용은 레이더 3의 +30 bin 쪽에 그대로 적용됩니다 — 87건 중 49%가 1차와 같은 호흡 주파수를 갖고 있습니다.']));

A(new Paragraph({children:[new PageBreak()]}));
A(H1('2.  동기화 — 실험 1·2로 한정했을 때'));
A(P('정박점을 운동(스쿼트) 구간이 아니라 실험 1의 마지막 회전 기준으로 바꾸면, 실험 3~7을 탐색하다 생기던 실패가 사라집니다. 레이더 3대가 각자 찾은 무호흡 위치와 BIOPAC과의 시차가 모두 정상 범위인지로 판정했습니다.'));
A(tbl([2000,1400,5626],['판정','인원','내용'],[
  ['확실','27명','레이더 2~3대가 같은 지점에서 무호흡을 잡고 offset도 정상'],
  ['경계','1명','S15_JKH — 1대만 잡힘. 다만 호흡수 오차는 0.21 BPM으로 양호'],
  ['불가','1명','S01_CMS — 본 녹화가 12.5초(501프레임)뿐'],
],{leftAll:true}));
A(GAP(80));
A(P([R('이전에 실험 2에서 빠졌던 '),B('S10_JKH · S18_LJH가 완전히 회복'),
  R('되고 S15_JKH도 살아났습니다. 이 셋은 무호흡을 못 찾은 것이 아니라 운동 구간을 찾다가 뒤쪽 실험으로 빠진 경우였습니다. offset도 −9.1 ~ +10.2초(중앙값 +1.6)로 물리적으로 타당한 범위에 들어옵니다.')]));

A(H1('3.  호흡 복원이 얼마나 맞는가'));
A(P('레이더에서 복원한 호흡수를 BIOPAC과 비교했습니다. 두 시계는 각자의 무호흡 중심으로 정렬했으므로 offset 추정이 필요 없습니다. 평소 호흡 · 느린 호흡 · 회복 호흡 · 운동 후 호흡 네 구간에서, 피험자 28명 × 4구간 = 112건입니다. 호흡수는 스펙트럼 정점을 포물선 보간해 연속값으로 추정했습니다(격자 양자화 제거).'));
A(img('fig_base.png',600,414,
  'A: 서브젝트별 오차. B: 잘 맞는 사례 — 네 구간 모두 일치. C: 안 맞는 사례 — 정답에 가까운 레이더가 매번 있는데도 중앙값이 엉뚱한 것을 고릅니다.'));
A(GAP(60));
A(tbl([2600,1700,1700,3026],['융합 방식','MAE (BPM)','1 BPM 이내','성격'],[
  ['레이더 1 단독','2.25','77%','—'],
  ['레이더 2 단독','1.99','73%','—'],
  ['레이더 3 단독','1.51','79%','단독으로는 가장 좋음'],
  ['3대 중앙값 (기준선)','1.08','85%','학습 없음 — 모델이 이겨야 할 값'],
  ['3대 최선 (이론적 상한)','0.32','95%','매 순간 옳은 레이더를 고른 경우'],
],{leftAll:true}));
A(GAP(120));
A(P([B('가장 좋은 레이더는 고정되어 있지 않습니다. '),
  R('112건에서 최선이었던 레이더를 세어 보면 1번 38회 · 2번 36회 · 3번 38회로 거의 균등합니다. "항상 3번을 쓴다" 같은 고정 규칙으로는 상한에 닿을 수 없고, '),
  B('시점마다 판단해야 합니다'),R(' — 이것이 모델에게 남는 일입니다.')]));
A(GAP(120));
A(tbl([2600,1900,1900,2626],['구간','MAE (중앙값 융합)','MAE (최선 상한)','BIOPAC 평균'],[
  ['평소 호흡','0.81','0.11','14.4 BPM'],
  ['느린 호흡','0.81','0.16','6.0 BPM'],
  ['회복 호흡','1.72','0.44','14.9 BPM'],
  ['운동 후 호흡','0.97','1.45','22.9 BPM'],
],{leftAll:true}));
A(GAP(140));
A(H2('서브젝트별 결과'));
A(P('오차는 네 구간에서 3대 중앙값 융합으로 얻은 호흡수와 BIOPAC의 차이를 평균한 값입니다.'));
(function(){
  const col=[1200,620,1180]; const cols=[...col,...col,...col];
  const head=['피험자','오차','판정','피험자','오차','판정','피험자','오차','판정'];
  const n=Math.ceil(ROWS.length/3); const out=[];
  for(let i=0;i<n;i++){
    const r=[];
    for(let k=0;k<3;k++){ const it=ROWS[i+k*n]; r.push(...(it?it:['','',''])); }
    out.push(r);
  }
  A(tbl(cols,head,out,{size:15}));
})();

A(H2('I/Q 보정과 복조 방식 — 후보를 늘리면 상한이 내려갑니다'));
A(P([R('지금 쓰는 방식은 I/Q 평면의 점들을 직선에 사영하는 것입니다. 변조가 작을 때는 문제없지만, 정면에서 크게 움직이면 원호를 직선에 누르는 셈이라 '),
  B('파형이 일그러지고 배음이 생깁니다'),
  R('. 실제로 2배음/기본파 비가 높은 상위 25%는 MAE 2.56 · 1 BPM 이내 64%인 반면, 하위 25%는 1.07 · 87%입니다. 반면 진폭 자체는 오차와 거의 무관했습니다(상관 −0.09).')]));
A(P('그래서 교과서적인 대안 두 가지를 붙여 비교했습니다 — 원을 적합해 중심을 잡고 위상을 직접 푸는 arctangent 복조, 그리고 타원을 적합해 I/Q 불균형까지 편 뒤 같은 처리를 하는 방식입니다.'));
A(GAP(60));
A(tbl([2600,1600,1600,3226],['복조 방식','MAE','1 BPM 이내','성격'],[
  ['선형 투영 (현재)','1.08','85%','전반적으로 가장 안정적'],
  ['원적합 + arctangent','2.69','76%','느린 호흡에서 압도적 (0.81 → 0.10)'],
  ['타원보정 + arctangent','2.86','77%','느린 호흡 0.09. 체동이 크면 무너짐'],
],{leftAll:true}));
A(GAP(80));
A(P([B('전체 평균만 보면 새 방식이 졌습니다. '),
  R('원·타원 적합이 체동에 취약해 운동 후 구간에서 크게 무너지기 때문입니다(0.97 → 5.29/6.63). 그러나 느린 호흡에서는 '),
  B('0.81 → 0.10으로 8배 좋아집니다'),R('. 즉 어느 하나가 항상 옳은 것이 아니라 상황마다 다릅니다.')]));
A(GAP(80));
A(note('그래서 후보를 늘리는 쪽이 맞습니다',[
  '레이더 3대 × 복조 3방식 = 9개 후보를 놓고, 매번 최선을 고를 수 있다면 MAE 0.16 · 1 BPM 이내 97%입니다.',
  '현재 1.08 → 레이더만 고를 때 0.32 → 후보를 늘리면 0.16. 정보는 계속 데이터 안에 있습니다.',
  '최선이었던 후보는 9개에 고르게 흩어집니다(19·16·15·15·13·11·9·8·6회). 고정 규칙으로는 닿을 수 없습니다.',
  '모델의 임무가 여기서 확정됩니다 — 후보를 만들어 두고, 매 시점 어느 것을 믿을지 고르는 것.']));

A(new Paragraph({children:[new PageBreak()]}));
A(H1('4.  피드백 대응 현황'));
A(tbl([2600,1500,4926],['지적 사항','현황','내용'],[
  ['I/Q 위상 보정','부분','파싱 오류는 수정 완료. I·Q 간 위상 불일치 보정은 미적용 — 다음 작업'],
  ['S01_CMS 녹화 이상','발견','본 녹화가 12.5초(501프레임)뿐입니다. 실험이 담기지 않았습니다 — 재촬영 또는 제외 확정 필요'],
  ['고스트 함께 사용','확인 완료','위아래 한 쌍은 I/Q. 진짜 멀티패스는 r3 +30 bin에 존재하며 활용 가치 있음'],
  ['환경 파일로 배경 제거','불가','전달받은 데이터에 빈 방 녹화가 없습니다(요약 참조). 대체법도 시험했으나 악화 — 아래 표'],
  ['1번·2번·왕복 3개 사용','부분','1번·2번은 사용 중. 왕복 구간은 아직 미사용'],
  ['잘 맞는/안 맞는 비교','완료','2장 참조'],
  ['BIOPAC 이상 피험자 확인','완료','25 BPM 초과 0명. S22_KJH만 대본과 어긋남'],
  ['MoRe-Fi 등 논문 확인','진행','MoRe-Fi (SenSys 2021) 확인. 두 번째 논문은 제목 확인 필요'],
],{leftAll:true}));

A(H2('환경 파일 없이 지울 수 있는가 — 시험 결과'));
A(P('빈 방 기준 없이 정적 클러터를 지우는 표준 방법(거리-시간 행렬의 지배적 특이성분 제거)을 25명 전원에 적용해 호흡수 오차를 다시 쟀습니다. 결과는 오히려 나빠집니다.'));
A(P('아래 수치는 호흡수 추정 방식을 개선하기 전(격자 양자화 상태)에 측정한 것이라 절대값은 3장과 다릅니다. 세 조건을 같은 방식으로 쟀으므로 비교 자체는 유효합니다.',{size:17}));
A(tbl([2600,1700,1900,2826],['처리','MAE (BPM)','해상도 이내','2.5 이내'],[
  ['없음 (당시 기준)','2.13','80%','84%'],
  ['SVD 1성분 제거','3.63','60%','66%'],
  ['SVD 2성분 제거','4.28','55%','59%'],
],{leftAll:true}));
A(GAP(80));
A(P([R('앉아서 호흡하는 녹화에서는 지배적 성분이 곧 사람의 가슴 반사입니다. 그것을 지우면 신호도 같이 사라집니다. '),
  B('역으로, 진짜 빈 방 기준이 필요한 이유이기도 합니다'),
  R(' — 빈 방 참조는 사람은 건드리지 않고 정적인 것만 뺄 수 있기 때문입니다.')]));

A(H2('덧붙임 — sync_tool_S02.m 190번 줄'));
A(P([R('주신 원본 코드의 복소 복원이 '),B('comp = d(:,1:2:end) + 1i*d(:,2:2:end)'),
  R(' 입니다. 1장에서 다룬 파싱 방식 A와 같습니다. 저희 쪽 도구도 이 줄을 그대로 받아 쓰고 있어서 함께 수정할 예정입니다. 선배님 쪽 파형 복원도 같은 영향을 받고 있을 수 있어 공유드립니다.')]));

A(new Paragraph({children:[new PageBreak()]}));
A(H1('5.  모델 1차 결과'));
A(P('기준선을 확정한 뒤 첫 모델을 학습했습니다. 과제는 "호흡을 읽는 것"이 아니라 "매 시점 어느 후보를 믿을지 고르는 것"으로 정의했습니다. 30초 창 784개(28명 × 4구간), 창마다 레이더 3대 × 복조 3방식 = 9개 후보를 만들고, 각 후보의 신뢰도를 SNN이 매겨 하나를 고릅니다. 피험자 분리 4-fold입니다.'));
A(img('fig_model.png',600,383,
  'A: 구간별 성능. B: 오차 누적분포. C: 무엇이 성능을 만드는가 — 절제 실험.'));
A(GAP(60));
A(tbl([3200,1600,1600,2626],['방식','MAE (BPM)','1 BPM 이내','비고'],[
  ['기준선 · 3대 중앙값 (학습 없음)','1.66','79%','—'],
  ['파형만 · 후보별 채점 (SNN)','1.79','80%','기준선을 넘지 못함'],
  ['파형만 · 9개 공동 채점 (SNN)','2.03','75%','합의를 스스로 배우지 못함'],
  ['신뢰도 특징만 (SNN 없음)','1.34','87%','5초 학습'],
  ['특징 + SNN (최종)','1.18','88%','파라미터 7,971 · 발화율 15%'],
  ['상한 · 매번 최선 후보','0.46','95%','—'],
],{leftAll:true}));
A(GAP(120));
A(note('솔직하게 적어둘 것 — 이득의 대부분은 SNN이 아니라 특징에서 나옵니다',[
  '레이더 파형만 주면 SNN은 기준선(1.66)을 넘지 못합니다(1.79). 9개를 한꺼번에 보게 해도 2.03으로 더 나빠집니다.',
  '반면 손으로 만든 신뢰도 특징(배음비·합의수·전후 드리프트 등)만으로 1.34가 나옵니다. SNN은 그 위에 0.16을 더할 뿐입니다.',
  '인코딩 5종(rate·step-forward·population·delta·direct)의 차이도 1.18~1.26으로 미미했습니다. 앞선 분류 과제에서 direct가 15%p 뒤처졌던 것과 대조됩니다 — SNN 가지의 기여 자체가 작기 때문으로 봅니다.',
  '해석: 30초·10 Hz의 1차원 파형 하나로는 작은 LIF 망이 배울 것이 부족합니다. 스펙트럼이나 여러 range bin을 함께 주는 등 입력을 넓히는 것이 다음 과제입니다.']));
A(GAP(120));
A(P([B('그래도 선택기 자체는 작동합니다. '),
  R('고른 후보 분포를 보면 9개에 흩어져 있고(최다 40%), 구간마다 취향이 다릅니다 — 느린 호흡에서는 arctangent 복조를 32% 고르는데 다른 구간에서는 10% 안팎입니다. arctangent가 느린 호흡에서 유리하다는 신호 특성을 그대로 배운 셈입니다. 흥미롭게도 '),
  B('실제 최선 후보를 정확히 고르는 비율은 23%에 불과'),
  R('한데도 성능이 오릅니다 — 최선을 맞히기보다 최악을 피하는 쪽으로 학습된 것으로 보입니다.')]));
A(GAP(100));
A(P([B('남은 문제는 운동 후 호흡입니다. '),
  R('평소 0.97 → 0.75, 느린 0.87 → 0.16, 회복 2.21 → 1.54로 모두 좋아지는데 운동 후만 2.58 → 2.29로 거의 그대로입니다. 이 구간의 상한도 1.00으로 다른 구간(0.06~0.62)보다 훨씬 높아, 후보 자체에 정답이 잘 들어 있지 않습니다. 체동이 큰 구간을 위한 별도 처리가 필요합니다.')]));

A(H1('6.  다음 단계'));
A(P([B('1. 환경 파일 요청. '),R('빈 방 녹화만 있으면 2차 봉우리 중 어디까지가 사람이고 어디부터가 가구인지 바로 갈립니다. 대체법이 통하지 않는 것이 확인됐으므로, 나머지 30명과 함께 전달 부탁드리는 것이 가장 우선입니다.')]));
A(P([B('2. I/Q 위상 불일치 보정. '),R('파형 복원 품질에 직접 영향을 줍니다.')]));
A(P([B('3. 과제를 호흡수·파형 복원으로 전환. '),R('현재 분류 과제보다 문헌 표준에 맞고, 이번에 정량화한 지표를 그대로 이어갈 수 있습니다.')]));
A(P([B('4. 왕복 구간 추가. '),R('체동 중 호흡 복원은 MoRe-Fi 계열이 다루는 문제와 정확히 겹칩니다.')]));
A(P([B('5. MATLAB sync 도구 교체. '),R('sync_tool_HAI_all.m · sync_batch_all.m 이 아직 옛 파싱이라 히트맵이 왜곡된 상태입니다.')]));

A(H1('7.  확인·요청드릴 사항'));
A(tbl([2400,1500,5126],['항목','종류','내용'],[
  ['환경 파일 (빈 방 녹화)','요청','전달받은 30명 폴더에 없습니다. 정적 클러터 제거에 필요하며, 대체법은 시험 결과 오히려 악화됐습니다(1장). 나머지 30명과 함께 부탁드립니다.'],
  ['안내 화면 위치','확인','실험 2에서 피험자가 레이더 2를 바라본 것이 맞다면, 레이더 1과 3은 각각 ∓45°로 대칭이어야 합니다. 그런데 오차가 1.95 vs 1.17로 다릅니다. 화면이 레이더 2 자리에 정확히 있었는지, 조금 옆이었는지 알려주시면 각도 모델 입력이 확정됩니다.'],
  ['S01_CMS 재촬영','확인','본 녹화가 12.5초(501프레임)뿐입니다. 재촬영 가능 여부 또는 제외 확정이 필요합니다.'],
  ['다음 촬영 시 공통 트리거','요청','레이더와 BIOPAC을 동시에 시작시키는 신호가 없어, 신호 내부 사건(무호흡·회전)으로 정렬하고 있습니다. 방어는 되지만 비표준이라 심사에서 질문이 나올 수 있습니다. 다음 촬영부터 공통 트리거를 넣으면 이 과정 전체가 필요 없어집니다.'],
  ['sync 도구 파싱 수정본','공유','sync_tool_S02.m 190번 줄의 복소 복원이 인터리브 방식입니다. 수정본을 저희가 만들어 공유드리겠습니다. 선배님 쪽 파형 복원도 같은 영향을 받고 있을 수 있습니다.'],
  ['두 번째 추천 논문','확인','MoRe-Fi는 확인했습니다. 함께 말씀하신 다른 한 편의 제목을 알려주시면 확인하겠습니다.'],
  ['S22_KJH 특이사항','확인','호흡수 오차가 6.96 BPM으로 가장 큽니다. BIOPAC은 정상이고 레이더 쪽 선택이 실패하는 사례입니다. 촬영 당시 자세나 벨트 착용에 특이사항이 있었는지 확인 부탁드립니다.'],
],{leftAll:true,size:16}));

const doc=new Document({
  styles:{default:{document:{run:{font:F,size:20,color:INK},
    paragraph:{spacing:{line:300,lineRule:LineRuleType.AUTO}}}}},
  sections:[{properties:{page:{margin:{top:1300,bottom:1300,left:1440,right:1440}}},
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,
      children:[new TextRun({children:[PageNumber.CURRENT],font:F,size:16,color:MUT})]})]})},
    children:body}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync('회의자료_파싱과호흡복원.docx',b);console.log('written',b.length);});
