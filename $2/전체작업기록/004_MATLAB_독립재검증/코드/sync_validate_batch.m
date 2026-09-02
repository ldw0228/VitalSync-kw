function sync_validate_batch(datasetRoot, outputRoot)
%SYNC_VALIDATE_BATCH Non-interactive validation of sync_tool_S02 logic.
% This copy does not modify the supplied MATLAB files or source data.

FPS = 40;
FL = 185;
if ~exist(outputRoot, 'dir'), mkdir(outputRoot); end

subjects = dir(fullfile(datasetRoot, 'S*_*'));
subjects = subjects([subjects.isdir]);
[~, order] = sort({subjects.name});
subjects = subjects(order);

results = repmat(struct('subject','', 'status','', 'valid_radars',[], ...
    'marker_count',NaN, 'offset_s',NaN, 'radar_duration_s',NaN, ...
    'biopac_duration_s',NaN, 'marker_biopac_s',[], 'marker_radar_s',[], ...
    'note',''), 0, 1);

for s = 1:numel(subjects)
    subject = subjects(s).name;
    row = struct('subject',subject, 'status','', 'valid_radars',[], ...
        'marker_count',NaN, 'offset_s',NaN, 'radar_duration_s',NaN, ...
        'biopac_duration_s',NaN, 'marker_biopac_s',[], 'marker_radar_s',[], ...
        'note','');
    try
        valid = [];
        comp = {};
        for k = 1:3
            radar = load_radar(fullfile(datasetRoot, subject, num2str(k)), FL);
            if ~isempty(radar)
                valid(end+1) = k; %#ok<AGROW>
                comp{end+1} = radar; %#ok<AGROW>
            end
        end
        row.valid_radars = valid;
        if isempty(comp)
            error('sync_validate:noRadar', 'No UWB datafloat files');
        end

        L = min(cellfun(@(x) size(x,1), comp));
        mot = zeros(L-1,1);
        for k = 1:numel(comp)
            comp{k} = comp{k}(1:L,:);
            mot = mot + sum(abs(diff(comp{k},1,1)),2);
        end
        mot = mot / numel(comp);
        mot_s = movmean(mot, round(0.5*FPS));
        tm = (0:L-2)'/FPS;

        files = dir(fullfile(datasetRoot, subject, 'BIOPAC', '**', '*.mat'));
        if isempty(files), error('sync_validate:noBiopac', 'No BIOPAC MAT-file'); end
        data = load(fullfile(files(1).folder, files(1).name));
        rsp = double(data.data(:,1));
        if isfield(data, 'isi'), fsb = 1000/double(data.isi(1)); else, fsb = 250; end
        tb = (0:numel(rsp)-1)'/fsb;

        threshold = 8.5;
        above = rsp > threshold;
        edges = diff([false; above; false]);
        onset = find(edges == 1);
        offs = find(edges == -1)-1;
        marker = zeros(numel(onset),1);
        for i = 1:numel(onset)
            segment = onset(i):offs(i);
            [~, peak] = max(rsp(segment));
            marker(i) = tb(segment(1)+peak-1);
        end
        marker = merge_close(marker, 4);

        motn = (mot_s-min(mot_s))/(max(mot_s)-min(mot_s)+eps);
        front = marker(marker < 300);
        if isempty(front), front = marker; end
        candidates = -12:0.1:12;
        scores = zeros(size(candidates));
        for i = 1:numel(candidates)
            rt = front-candidates(i);
            rt = rt(rt > 0 & rt < tm(end));
            if ~isempty(rt)
                scores(i) = mean(interp1(tm, motn, rt, 'linear', 0));
            end
        end
        [~, best] = max(scores);
        offset = candidates(best);

        row.status = 'OK';
        row.marker_count = numel(marker);
        row.offset_s = offset;
        row.radar_duration_s = (L-1)/FPS;
        row.biopac_duration_s = tb(end);
        row.marker_biopac_s = marker(:)';
        row.marker_radar_s = (marker-offset)';
        if numel(valid) < 3, row.note = sprintf('available radars: %s', mat2str(valid)); end
    catch err
        row.status = 'FAILED';
        row.note = sprintf('%s: %s', err.identifier, err.message);
    end
    results(end+1,1) = row; %#ok<AGROW>
    fprintf('%s: %s markers=%g offset=%+.1f\n', subject, row.status, row.marker_count, row.offset_s);
end

save(fullfile(outputRoot, 'matlab_results.mat'), 'results');
fid = fopen(fullfile(outputRoot, 'matlab_results.json'), 'w', 'n', 'UTF-8');
fwrite(fid, jsonencode(results, PrettyPrint=true), 'char');
fclose(fid);

summary = table(string({results.subject})', string({results.status})', ...
    [results.marker_count]', [results.offset_s]', ...
    cellfun(@numel, {results.valid_radars})', string({results.note})', ...
    'VariableNames', {'subject','status','marker_count','offset_s','radar_count','note'});
writetable(summary, fullfile(outputRoot, 'matlab_summary.csv'), 'Encoding', 'UTF-8');
end


function marker = merge_close(marker, seconds)
if isempty(marker), return; end
marker = sort(marker(:));
group = marker(1);
merged = [];
for i = 2:numel(marker)
    if marker(i)-group(end) <= seconds
        group(end+1) = marker(i); %#ok<AGROW>
    else
        merged(end+1,1) = mean(group); %#ok<AGROW>
        group = marker(i);
    end
end
merged(end+1,1) = mean(group);
marker = merged;
end


function comp = load_radar(folder, frameLength)
files = dir(fullfile(folder, '**', 'xethru_datafloat_*.dat'));
if isempty(files), comp = []; return; end
[~, order] = sort({files.name});
files = files(order);
parts = {};
for i = 1:numel(files)
    fid = fopen(fullfile(files(i).folder, files(i).name), 'r');
    raw = fread(fid, inf, 'float32=>double', 0, 'ieee-le');
    fclose(fid);
    frames = floor(numel(raw)/frameLength);
    if frames < 40, continue; end
    matrix = reshape(raw(1:frames*frameLength), frameLength, frames)';
    parts{end+1} = matrix(:,2:end); %#ok<AGROW>
end
if isempty(parts), comp = []; return; end
data = cat(1, parts{:});
comp = data(:,1:2:end) + 1i*data(:,2:2:end);
end
