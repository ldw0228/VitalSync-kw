function sync_tool_S02()
%% ========================================================================
%  S02_RJS  UWB-BIOPAC 인터랙티브 sync 도구
%  - BIOPAC 흉부누름 마커(강한 입력) 자동 검출 + 앞 마커로 자동 정렬
%  - UWB heatmap(3레이더)에 이벤트 구간 분할선
%  - 슬라이더로 shift → 창 자동 업데이트 / [저장] 버튼으로 shift된 BIOPAC 저장
%  VSCode MATLAB extension에서 Run.  (Signal Processing / Statistics Toolbox 불필요)
%  ========================================================================
    % Configure these values for the local dataset.
    SUBJ  = 'SUBJECT_CODE';
    RAW   = 'PATH_TO_RAW_DATA';
    OUTD  = 'PATH_TO_SYNC_OUTPUT';
    FPS   = 40;  FL = 185;  NB = 92;
    if ~exist(OUTD,'dir'); mkdir(OUTD); end

    %% ---- 레이더 3대 로드 (heatmap + 움직임) ----
    fprintf('레이더 로드 중...\n');
    comp = cell(1,3);
    for k = 1:3
        comp{k} = load_radar(fullfile(RAW,SUBJ,num2str(k)), FL);
    end
    L = min(cellfun(@(c) size(c,1), comp));
    for k = 1:3, comp{k} = comp{k}(1:L,:); end
    t_rad = (0:L-1)/FPS;
    mag = cell(1,3); mot = zeros(L-1,1);
    for k = 1:3
        cr = comp{k} - mean(comp{k},1);
        mag{k} = abs(cr);
        mot = mot + sum(abs(diff(comp{k},1,1)),2);
    end
    mot = mot/3;  mot_s = movmean(mot, round(0.5*FPS));
    tm  = (0:L-2)/FPS;

    %% ---- BIOPAC RSP 로드 ----
    f = dir(fullfile(RAW,SUBJ,'BIOPAC','**','*.mat'));
    S = load(fullfile(f(1).folder, f(1).name));
    rsp = double(S.data(:,1));
    if isfield(S,'isi'), fsb = 1000/double(S.isi(1)); else, fsb = 250; end
    tb = (0:numel(rsp)-1)'/fsb;

    %% ---- 흉부누름 마커 검출 (RSP 상단 튐: 호흡 ~2.5~7.5, 누름 ~10) ----
    thr = 8.5;
    above = rsp > thr;
    dd = diff([0; above; 0]);
    onset = find(dd==1); offs = find(dd==-1)-1;
    mk_bio = [];
    for i = 1:numel(onset)
        seg = onset(i):offs(i);
        [~,mx] = max(rsp(seg));  mk_bio(end+1,1) = tb(seg(1)+mx-1); %#ok<AGROW>
    end
    if ~isempty(mk_bio)                              % 4초 이내 병합
        mk_bio = sort(mk_bio); grp = mk_bio(1); merged = [];
        for i = 2:numel(mk_bio)
            if mk_bio(i)-grp(end) <= 4, grp(end+1)=mk_bio(i); else, merged(end+1)=mean(grp); grp=mk_bio(i); end %#ok<AGROW>
        end
        merged(end+1) = mean(grp); mk_bio = merged(:);
    end
    fprintf('흉부누름 마커 %d개 검출\n', numel(mk_bio));

    %% ---- 자동 offset: 앞구간(0~300s) 마커가 레이더 움직임에 얹히도록 ----
    %  정의: biopac_time = radar_time + offset  ->  radar_time = biopac_time - offset
    motn = (mot_s - min(mot_s)) / (max(mot_s)-min(mot_s)+eps);
    front = mk_bio(mk_bio < 300);
    if isempty(front), front = mk_bio; end
    cand = -12:0.1:12;  score = zeros(size(cand));
    for i = 1:numel(cand)
        rt = front - cand(i);  rt = rt(rt>0 & rt<tm(end));
        if isempty(rt), continue; end
        score(i) = mean(interp1(tm, motn, rt, 'linear', 0));
    end
    [~,bi] = max(score);  offset0 = cand(bi);
    fprintf('자동 offset = %+.2f s\n', offset0);

    %% ---- Figure 구성 ----
    fig = figure('Name',['sync tool - ' SUBJ], 'Position',[40 40 1650 950], 'Color','w');
    ds = 8;                                          % heatmap 표시 다운샘플
    axR = subplot(5,1,1);                            % (1) BIOPAC RSP + 마커
    plot(axR, tb, rsp, 'Color',[0 .5 0], 'LineWidth',0.4); hold(axR,'on');
    yl = ylim(axR);
    for i = 1:numel(mk_bio)
        plot(axR, [mk_bio(i) mk_bio(i)], yl, 'r--','LineWidth',1);
        text(axR, mk_bio(i), yl(2), sprintf(' %.0f', mk_bio(i)), 'Color','r','FontSize',7,'VerticalAlignment','top');
    end
    title(axR, sprintf('BIOPAC RSP + 흉부누름 마커 %d개 (biopac 시간)', numel(mk_bio)));
    ylabel(axR,'V'); xlim(axR,[0 tb(end)]); grid(axR,'on');

    axH = gobjects(3,1);                             % (2~4) radar heatmap + 경계선
    for k = 1:3
        axH(k) = subplot(5,1,k+1);
        imagesc(axH(k), t_rad(1:ds:end), 1:NB, mag{k}(1:ds:end,:)');
        set(axH(k),'YDir','normal'); colormap(axH(k),'hot');
        clim(axH(k), [0 pctl(mag{k}(:),99.5)]);
        ylabel(axH(k), sprintf('r%d bin',k)); hold(axH(k),'on');
        title(axH(k), sprintf('radar%d heatmap (하늘선=이벤트 경계)',k));
        xlim(axH(k),[0 t_rad(end)]);
    end

    subplot(5,1,5); axis off;                        % (5) 컨트롤
    txt = uicontrol('Style','text','Units','normalized','Position',[0.13 0.10 0.40 0.03], ...
        'String',sprintf('offset = %+.2f s', offset0),'FontSize',11,'BackgroundColor','w','HorizontalAlignment','left');
    sld = uicontrol('Style','slider','Units','normalized','Position',[0.13 0.06 0.55 0.03], ...
        'Min',offset0-15,'Max',offset0+15,'Value',offset0,'SliderStep',[0.1/30 1/30]);
    btnAuto = uicontrol('Style','pushbutton','Units','normalized','Position',[0.70 0.05 0.09 0.045], ...
        'String','자동값 복귀','FontSize',10);
    btnSave = uicontrol('Style','pushbutton','Units','normalized','Position',[0.80 0.05 0.10 0.045], ...
        'String','shift 저장','FontSize',10,'FontWeight','bold','BackgroundColor',[.8 .95 .8]);

    %% ---- 상태 저장 (guidata) ----
    st = struct();
    st.SUBJ = SUBJ;  st.OUTD = OUTD;  st.FPS = FPS;  st.NB = NB;
    st.mk_bio = mk_bio;  st.offset0 = offset0;  st.curOffset = offset0;
    st.axH = axH;  st.hEv = {gobjects(0),gobjects(0),gobjects(0)};  st.txt = txt;  st.sld = sld;
    st.tb = tb;  st.rsp = rsp;  st.t_rad = t_rad(:);
    guidata(fig, st);

    drawEvents(fig);                                 % 초기 이벤트선
    linkaxes(axH,'x');

    %% ---- 콜백 연결 (local function 참조) ----
    addlistener(sld, 'Value', 'PostSet', @(s,e) onSlide(fig));   % 실시간 업데이트
    set(btnAuto,'Callback', @(s,e) resetAuto(fig));
    set(btnSave,'Callback', @(s,e) saveShift(fig));
end

%% ======================= 로컬 함수 (콜백) =============================
function drawEvents(fig)
    st = guidata(fig);
    xe = st.mk_bio - st.curOffset;                   % 마커 -> 레이더축(이벤트 경계)
    for k = 1:3
        old = st.hEv{k};
        if ~isempty(old); delete(old(ishghandle(old))); end
        hh = gobjects(numel(xe),1);
        for i = 1:numel(xe)
            hh(i) = plot(st.axH(k), [xe(i) xe(i)], [0 st.NB], 'c--','LineWidth',1.2);
        end
        st.hEv{k} = hh;
    end
    set(st.txt,'String',sprintf('offset = %+.2f s   (biopac_t = radar_t + offset)', st.curOffset));
    guidata(fig, st);
end

function onSlide(fig)
    st = guidata(fig);
    st.curOffset = get(st.sld,'Value');
    guidata(fig, st);
    drawEvents(fig);                                 % 창 자동 업데이트
    drawnow limitrate;
end

function resetAuto(fig)
    st = guidata(fig);
    st.curOffset = st.offset0;  set(st.sld,'Value',st.offset0);
    guidata(fig, st);
    drawEvents(fig);  drawnow;
end

function saveShift(fig)
    st = guidata(fig);
    % biopac_time = radar_time + curOffset  ->  레이더축으로 RSP 리샘플 정렬
    rsp_on_radar = interp1(st.tb - st.curOffset, st.rsp, st.t_rad, 'linear', NaN);
    out = struct('subject',st.SUBJ,'offset_s',st.curOffset,'fs_radar',st.FPS, ...
        't_radar',st.t_rad,'rsp_aligned',rsp_on_radar, ...
        'marker_radar_s', st.mk_bio - st.curOffset, 'marker_biopac_s', st.mk_bio, ...
        'note','RSP shifted to radar time by offset (biopac_t=radar_t+offset)');
    fn = fullfile(st.OUTD, sprintf('%s_biopac_shift.mat', st.SUBJ));
    save(fn, '-struct', 'out');
    fprintf('저장 완료: %s  (offset %+.2f s)\n', fn, st.curOffset);
    msgbox(sprintf('저장 완료\n%s\noffset %+.2f s', fn, st.curOffset), 'saved');
end

%% ======================= 로컬 함수 (유틸) ============================
function v = pctl(x, p)
    x = sort(x(~isnan(x)));
    if isempty(x), v = 1; return; end
    v = x(max(1, min(numel(x), round(p/100*numel(x)))));
end

function comp = load_radar(folder, FL)
    f = dir(fullfile(folder,'**','xethru_datafloat_*.dat'));
    [~,ord] = sort({f.name}); f = f(ord);
    parts = {};
    for i = 1:numel(f)
        fid = fopen(fullfile(f(i).folder, f(i).name),'r');
        raw = fread(fid, inf, 'float32'); fclose(fid);
        n = floor(numel(raw)/FL);
        if n < 40, continue; end
        m = reshape(raw(1:n*FL), FL, n).';
        parts{end+1} = m(:,2:end); %#ok<AGROW>
    end
    d = cat(1, parts{:});
    comp = d(:,1:2:end) + 1i*d(:,2:2:end);
end
