clear; clc; close all;

workspace_dir = 'C:\Users\rkdeh\Documents\Codex\2026-07-01\d-uwb';
data_dir = 'D:\UWB\UWB_BIOPAC_DATA_0601';
out_dir = fullfile(workspace_dir, 'outputs', 'shift_analysis');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fs_uwb_default = 17;
fs_bio_default = 250;
window_sec = 30;
stride_sec = 15;
freq_band_bpm = [6 40];
time_shifts = -10:0.5:10;
bin_shifts = -20:20;

files = dir(fullfile(data_dir, 'SyncData_sub*.mat'));

time_rows = strings(0, 12);
range_rows = strings(0, 13);
file_rows = strings(0, 11);

for fi = 1:numel(files)
    file_path = fullfile(files(fi).folder, files(fi).name);
    file_status = ["", "", "", "", "", "", "", "", "", "", ""];
    file_status(1) = string(files(fi).name);
    try
        loaded = load(file_path, 'SyncData');
        if ~isfield(loaded, 'SyncData')
            file_status(2) = "skipped";
            file_status(11) = "No SyncData field";
            file_rows(end+1,:) = file_status; %#ok<AGROW>
            continue;
        end
        s = loaded.SyncData;
        if ~has_required_fields(s)
            file_status(2) = "skipped";
            file_status(11) = "Missing required fields";
            if isfield(s, 'subject_num'), file_status(3) = string(double(s.subject_num)); end
            if isfield(s, 'subject_name'), file_status(4) = string(s.subject_name); end
            file_rows(end+1,:) = file_status; %#ok<AGROW>
            continue;
        end

        subject_num = double(s.subject_num);
        subject_name = string(s.subject_name);
        fs_uwb = get_field_or_default(s, 'Fs_uwb', fs_uwb_default);
        fs_bio = get_field_or_default(s, 'Fs_biopac', fs_bio_default);

        com = double(s.com_final);
        tv = double(s.tv_final);
        n_bins = min(size(com, 1), size(tv, 1));
        n_frames = min(size(com, 2), size(tv, 2));
        com = com(1:n_bins, 1:n_frames);
        tv = tv(1:n_bins, 1:n_frames);
        [com_z, tv_z] = robust_subject_zscore(com, tv);

        if isfield(s, 'time_final')
            uwb_time = double(s.time_final(:)');
            uwb_time = uwb_time(1:min(numel(uwb_time), n_frames));
        else
            uwb_time = (0:n_frames-1) / fs_uwb;
        end
        n_frames = min(n_frames, numel(uwb_time));
        com_z = com_z(:, 1:n_frames);
        tv_z = tv_z(:, 1:n_frames);

        start_times = uwb_time(1):stride_sec:(min(uwb_time(end), max(s.bio_time_final)) - window_sec);
        windows_used = 0;

        for wi = 1:numel(start_times)
            st_time = start_times(wi);
            ed_time = st_time + window_sec;
            bio_idx = get_time_indices(s.bio_time_final, st_time, ed_time);
            if numel(bio_idx) < round(window_sec * fs_bio * 0.8)
                continue;
            end
            rr_bio = estimate_rr_fft(double(s.bpf_bio_final(bio_idx)), fs_bio, freq_band_bpm);
            if ~isfinite(rr_bio)
                continue;
            end

            base_peak_com = choose_peak_bin(s, "com", st_time, ed_time, com_z, uwb_time);
            base_peak_tv = choose_peak_bin(s, "tv", st_time, ed_time, tv_z, uwb_time);

            % Time-shift sweep on the existing 1D UWB BPF respiration signals.
            for si = 1:numel(time_shifts)
                shift = time_shifts(si);
                ust = st_time + shift;
                ued = ed_time + shift;
                u_idx = get_time_indices(s.uwb_time, ust, ued);
                if numel(u_idx) < round(window_sec * fs_uwb * 0.8)
                    continue;
                end
                rr_com = estimate_rr_fft(double(s.bpf_com(u_idx)), fs_uwb, freq_band_bpm);
                rr_tv = estimate_rr_fft(double(s.bpf_tv(u_idx)), fs_uwb, freq_band_bpm);
                time_rows(end+1,:) = [ ...
                    string(files(fi).name), string(subject_num), subject_name, string(wi), ...
                    string(st_time), string(ed_time), string(shift), string(rr_bio), ...
                    string(rr_com), string(rr_tv), string(abs(rr_com - rr_bio)), string(abs(rr_tv - rr_bio)) ...
                ]; %#ok<AGROW>
            end

            % Range-bin shift sweep around the estimated person/respiration peak bin.
            r_idx = get_time_indices(uwb_time, st_time, ed_time);
            if numel(r_idx) >= round(window_sec * fs_uwb * 0.8)
                for bi = 1:numel(bin_shifts)
                    bin_shift = bin_shifts(bi);
                    com_bin = min(max(base_peak_com + bin_shift, 1), n_bins);
                    tv_bin = min(max(base_peak_tv + bin_shift, 1), n_bins);
                    rr_com_bin = estimate_rr_fft(com_z(com_bin, r_idx), fs_uwb, freq_band_bpm);
                    rr_tv_bin = estimate_rr_fft(tv_z(tv_bin, r_idx), fs_uwb, freq_band_bpm);
                    range_rows(end+1,:) = [ ...
                        string(files(fi).name), string(subject_num), subject_name, string(wi), ...
                        string(st_time), string(ed_time), string(bin_shift), string(base_peak_com), ...
                        string(base_peak_tv), string(rr_bio), string(rr_com_bin), string(rr_tv_bin), ...
                        string(abs(rr_com_bin - rr_bio)) ...
                    ]; %#ok<AGROW>
                end
            end
            windows_used = windows_used + 1;
        end

        file_status = [string(files(fi).name), "used", string(subject_num), subject_name, ...
            string(windows_used), string(numel(start_times)), string(n_frames), string(n_bins), ...
            string(min(time_shifts)), string(max(time_shifts)), "ok"];
        file_rows(end+1,:) = file_status; %#ok<AGROW>
    catch ME
        file_status(2) = "error";
        file_status(11) = string(ME.message);
        file_rows(end+1,:) = file_status; %#ok<AGROW>
    end
end

time_table = array2table(time_rows, 'VariableNames', ...
    {'file','subject_num','subject_name','window_index','start_sec','end_sec','time_shift_sec', ...
    'rr_bio','rr_com','rr_tv','ae_com','ae_tv'});
range_table = array2table(range_rows, 'VariableNames', ...
    {'file','subject_num','subject_name','window_index','start_sec','end_sec','bin_shift', ...
    'base_peak_com','base_peak_tv','rr_bio','rr_com_bin','rr_tv_bin','ae_com_bin'});
file_table = array2table(file_rows, 'VariableNames', ...
    {'file','status','subject_num','subject_name','windows_used','candidate_windows','frames','bins', ...
    'min_time_shift','max_time_shift','message'});

writetable(time_table, fullfile(out_dir, 'time_shift_window_results.csv'));
writetable(range_table, fullfile(out_dir, 'range_shift_window_results.csv'));
writetable(file_table, fullfile(out_dir, 'shift_dataset_build_summary.csv'));

time_summary = summarize_time_shift(time_table, time_shifts);
range_summary = summarize_range_shift(range_table, bin_shifts);
writetable(time_summary, fullfile(out_dir, 'time_shift_summary.csv'));
writetable(range_summary, fullfile(out_dir, 'range_shift_summary.csv'));

make_shift_plots(out_dir, time_summary, range_summary);
write_shift_report(out_dir, time_summary, range_summary, file_table, window_sec, stride_sec, freq_band_bpm);

disp(time_summary);
disp(range_summary);
fprintf('Outputs written to %s\n', out_dir);

function ok = has_required_fields(s)
    ok = isfield(s, 'com_final') && isfield(s, 'tv_final') && ...
        isfield(s, 'bpf_bio_final') && isfield(s, 'bio_time_final') && ...
        isfield(s, 'bpf_com') && isfield(s, 'bpf_tv') && isfield(s, 'uwb_time') && ...
        isfield(s, 'subject_num') && isfield(s, 'subject_name');
end

function value = get_field_or_default(s, field, default_value)
    if isfield(s, field)
        value = double(s.(field));
    else
        value = default_value;
    end
end

function idx = get_time_indices(t, st_time, ed_time)
    t = double(t(:)');
    idx = find(t >= st_time & t < ed_time);
end

function [com_z, tv_z] = robust_subject_zscore(com, tv)
    com_center = median(com, 2, 'omitnan');
    tv_center = median(tv, 2, 'omitnan');
    com_scale = median(abs(com - com_center), 2, 'omitnan');
    tv_scale = median(abs(tv - tv_center), 2, 'omitnan');
    com_scale(com_scale < 1e-6) = 1;
    tv_scale(tv_scale < 1e-6) = 1;
    com_z = (com - com_center) ./ com_scale;
    tv_z = (tv - tv_center) ./ tv_scale;
    com_z(~isfinite(com_z)) = 0;
    tv_z(~isfinite(tv_z)) = 0;
end

function peak_bin = choose_peak_bin(s, kind, st_time, ed_time, signal, signal_time)
    peak_field = "snr_peak_bin_" + kind;
    centers_field = "snr_window_centers";
    if isfield(s, peak_field) && isfield(s, centers_field)
        centers = double(s.(centers_field)(:));
        peaks = double(s.(peak_field)(:));
        mask = centers >= st_time & centers < ed_time & isfinite(peaks);
        if any(mask)
            peak_bin = round(median(peaks(mask), 'omitnan'));
            peak_bin = min(max(peak_bin, 1), size(signal, 1));
            return;
        end
    end
    idx = get_time_indices(signal_time, st_time, ed_time);
    if isempty(idx)
        peak_bin = round(size(signal, 1) / 2);
    else
        [~, peak_bin] = max(mean(abs(diff(signal(:, idx), 1, 2)), 2, 'omitnan'));
    end
end

function rr = estimate_rr_fft(signal, fs, band_bpm)
    signal = double(signal(:));
    signal = signal(isfinite(signal));
    if numel(signal) < fs * 8 || std(signal) < 1e-9
        rr = NaN;
        return;
    end
    signal = signal - mean(signal);
    n = numel(signal);
    win = 0.5 - 0.5*cos(2*pi*(0:n-1)'/max(n-1,1));
    signal = signal .* win;
    nfft = 2^nextpow2(max(n, 4096));
    spectrum = abs(fft(signal, nfft)).^2;
    freqs = (0:nfft-1)' * fs / nfft;
    band_hz = band_bpm / 60;
    mask = freqs >= band_hz(1) & freqs <= band_hz(2);
    if ~any(mask)
        rr = NaN;
        return;
    end
    band_power = spectrum(mask);
    band_freqs = freqs(mask);
    [~, idx] = max(band_power);
    rr = band_freqs(idx) * 60;
end

function summary = summarize_time_shift(time_table, shifts)
    rows = strings(0, 8);
    shift_col = numeric_column(time_table.time_shift_sec);
    rr_bio = numeric_column(time_table.rr_bio);
    rr_com = numeric_column(time_table.rr_com);
    rr_tv = numeric_column(time_table.rr_tv);
    for si = 1:numel(shifts)
        mask = abs(shift_col - shifts(si)) < 1e-9;
        ae_com = abs(rr_com - rr_bio);
        ae_tv = abs(rr_tv - rr_bio);
        rows(end+1,:) = ["COM", string(shifts(si)), string(sum(mask & isfinite(ae_com))), ...
            string(mean(ae_com(mask), 'omitnan')), string(median(ae_com(mask), 'omitnan')), ...
            string(simple_corr(rr_com(mask), rr_bio(mask))), string(mean(rr_com(mask), 'omitnan')), string(mean(rr_bio(mask), 'omitnan'))]; %#ok<AGROW>
        rows(end+1,:) = ["TV", string(shifts(si)), string(sum(mask & isfinite(ae_tv))), ...
            string(mean(ae_tv(mask), 'omitnan')), string(median(ae_tv(mask), 'omitnan')), ...
            string(simple_corr(rr_tv(mask), rr_bio(mask))), string(mean(rr_tv(mask), 'omitnan')), string(mean(rr_bio(mask), 'omitnan'))]; %#ok<AGROW>
    end
    summary = array2table(rows, 'VariableNames', ...
        {'source','time_shift_sec','n','mae_bpm','median_ae_bpm','corr_with_bio','mean_rr_uwb','mean_rr_bio'});
end

function summary = summarize_range_shift(range_table, bin_shifts)
    rows = strings(0, 7);
    shift_col = numeric_column(range_table.bin_shift);
    rr_bio = numeric_column(range_table.rr_bio);
    rr_com = numeric_column(range_table.rr_com_bin);
    rr_tv = numeric_column(range_table.rr_tv_bin);
    for si = 1:numel(bin_shifts)
        mask = shift_col == bin_shifts(si);
        ae_com = abs(rr_com - rr_bio);
        ae_tv = abs(rr_tv - rr_bio);
        rows(end+1,:) = ["COM", string(bin_shifts(si)), string(sum(mask & isfinite(ae_com))), ...
            string(mean(ae_com(mask), 'omitnan')), string(median(ae_com(mask), 'omitnan')), ...
            string(simple_corr(rr_com(mask), rr_bio(mask))), string(mean(rr_com(mask), 'omitnan'))]; %#ok<AGROW>
        rows(end+1,:) = ["TV", string(bin_shifts(si)), string(sum(mask & isfinite(ae_tv))), ...
            string(mean(ae_tv(mask), 'omitnan')), string(median(ae_tv(mask), 'omitnan')), ...
            string(simple_corr(rr_tv(mask), rr_bio(mask))), string(mean(rr_tv(mask), 'omitnan'))]; %#ok<AGROW>
    end
    summary = array2table(rows, 'VariableNames', ...
        {'source','bin_shift','n','mae_bpm','median_ae_bpm','corr_with_bio','mean_rr_uwb'});
end

function values = numeric_column(col)
    if isnumeric(col)
        values = double(col);
    else
        values = str2double(string(col));
    end
end

function r = simple_corr(x, y)
    x = double(x(:));
    y = double(y(:));
    mask = isfinite(x) & isfinite(y);
    x = x(mask);
    y = y(mask);
    if numel(x) < 2 || std(x) < 1e-12 || std(y) < 1e-12
        r = NaN;
        return;
    end
    x = x - mean(x);
    y = y - mean(y);
    r = sum(x .* y) / sqrt(sum(x.^2) * sum(y.^2));
end

function make_shift_plots(out_dir, time_summary, range_summary)
    fig = figure('Visible', 'off', 'Position', [100 100 1100 700]);
    tiledlayout(1, 2, 'TileSpacing', 'compact');
    nexttile; hold on;
    src = string(time_summary.source);
    shifts = numeric_column(time_summary.time_shift_sec);
    mae = numeric_column(time_summary.mae_bpm);
    for s = ["COM", "TV"]
        mask = src == s;
        plot(shifts(mask), mae(mask), '-o', 'LineWidth', 1.5, 'DisplayName', s);
    end
    xlabel('UWB time shift vs BIOPAC window (sec)');
    ylabel('RR MAE vs BIOPAC (bpm)');
    title('Time shift sweep');
    grid on; legend('Location', 'best');

    nexttile; hold on;
    src = string(range_summary.source);
    shifts = numeric_column(range_summary.bin_shift);
    mae = numeric_column(range_summary.mae_bpm);
    for s = ["COM", "TV"]
        mask = src == s;
        plot(shifts(mask), mae(mask), '-o', 'LineWidth', 1.5, 'DisplayName', s);
    end
    xlabel('Range-bin shift around peak bin');
    ylabel('RR MAE vs BIOPAC (bpm)');
    title('Range-bin shift sweep');
    grid on; legend('Location', 'best');
    exportgraphics(fig, fullfile(out_dir, 'shift_sweep_summary.png'), 'Resolution', 160);
    close(fig);
end

function write_shift_report(out_dir, time_summary, range_summary, file_table, window_sec, stride_sec, freq_band_bpm)
    report_path = fullfile(out_dir, 'shift_analysis_summary_ko.md');
    fid = fopen(report_path, 'w', 'n', 'UTF-8');
    cleanup = onCleanup(@() fclose(fid));

    time_mae = numeric_column(time_summary.mae_bpm);
    range_mae = numeric_column(range_summary.mae_bpm);
    [best_time_mae, best_time_idx] = min(time_mae);
    [best_range_mae, best_range_idx] = min(range_mae);
    zero_com = time_summary(string(time_summary.source) == "COM" & numeric_column(time_summary.time_shift_sec) == 0, :);
    zero_tv = time_summary(string(time_summary.source) == "TV" & numeric_column(time_summary.time_shift_sec) == 0, :);
    zero_bin_com = range_summary(string(range_summary.source) == "COM" & numeric_column(range_summary.bin_shift) == 0, :);
    zero_bin_tv = range_summary(string(range_summary.source) == "TV" & numeric_column(range_summary.bin_shift) == 0, :);

    used = sum(string(file_table.status) == "used");
    skipped = sum(string(file_table.status) ~= "used");

    fprintf(fid, '# UWB-BIOPAC Shift Analysis\n\n');
    fprintf(fid, '## 한 줄 결론\n\n');
    fprintf(fid, 'BIOPAC RR 기준으로 시간 shift와 range-bin shift를 검사했다. 시간 shift best는 `%s` %+g초에서 MAE %.2f bpm이고, range-bin shift best는 `%s` %+g bin에서 MAE %.2f bpm이다.\n\n', ...
        string(time_summary.source(best_time_idx)), str2double(string(time_summary.time_shift_sec(best_time_idx))), best_time_mae, ...
        string(range_summary.source(best_range_idx)), str2double(string(range_summary.bin_shift(best_range_idx))), best_range_mae);

    fprintf(fid, '## 설정\n\n');
    fprintf(fid, '- Window: %d초\n', window_sec);
    fprintf(fid, '- Stride: %d초\n', stride_sec);
    fprintf(fid, '- RR 탐색 대역: %.0f-%.0f bpm\n', freq_band_bpm(1), freq_band_bpm(2));
    fprintf(fid, '- Time shift sweep: -10초부터 +10초까지 0.5초 간격\n');
    fprintf(fid, '- Range-bin shift sweep: peak bin 주변 -20부터 +20 bin\n');
    fprintf(fid, '- 사용 파일: %d개, 제외/오류 파일: %d개\n\n', used, skipped);

    fprintf(fid, '## 기준점 대비 개선 여부\n\n');
    fprintf(fid, '| Test | Source | Baseline shift | Baseline MAE | Best shift | Best MAE | Delta |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---:|---:|\n');
    fprintf(fid, '| Time | COM | 0 sec | %.2f | %.1f sec | %.2f | %.2f |\n', ...
        str2double(string(zero_com.mae_bpm(1))), str2double(string(time_summary.time_shift_sec(best_time_idx))), best_time_mae, ...
        best_time_mae - str2double(string(zero_com.mae_bpm(1))));
    fprintf(fid, '| Time | TV | 0 sec | %.2f | %.1f sec | %.2f | %.2f |\n', ...
        str2double(string(zero_tv.mae_bpm(1))), str2double(string(time_summary.time_shift_sec(best_time_idx))), best_time_mae, ...
        best_time_mae - str2double(string(zero_tv.mae_bpm(1))));
    fprintf(fid, '| Range | COM | 0 bin | %.2f | %+d bin | %.2f | %.2f |\n', ...
        str2double(string(zero_bin_com.mae_bpm(1))), round(str2double(string(range_summary.bin_shift(best_range_idx)))), best_range_mae, ...
        best_range_mae - str2double(string(zero_bin_com.mae_bpm(1))));
    fprintf(fid, '| Range | TV | 0 bin | %.2f | %+d bin | %.2f | %.2f |\n\n', ...
        str2double(string(zero_bin_tv.mae_bpm(1))), round(str2double(string(range_summary.bin_shift(best_range_idx)))), best_range_mae, ...
        best_range_mae - str2double(string(zero_bin_tv.mae_bpm(1))));

    fprintf(fid, '## 해석\n\n');
    fprintf(fid, '- Time shift는 UWB와 BIOPAC의 동기화 오차가 남아 있는지 확인하는 검사다.\n');
    fprintf(fid, '- Range-bin shift는 peak bin 주변에서 호흡이 가장 잘 잡히는 위치가 실제 peak와 어긋나는지 확인하는 검사다.\n');
    fprintf(fid, '- best shift가 0 근처면 기존 sync/peak 선택이 대체로 맞다는 뜻이고, 0에서 멀면 보정 여지가 있다는 뜻이다.\n');
    fprintf(fid, '- 단, RR을 30초 FFT dominant frequency로 뽑았기 때문에 shift에 따른 변화가 작게 보일 수 있다. 파형 correlation 기반 shift 검사를 추가하면 더 민감하게 볼 수 있다.\n\n');

    fprintf(fid, '## 생성 파일\n\n');
    fprintf(fid, '- `time_shift_summary.csv`\n');
    fprintf(fid, '- `range_shift_summary.csv`\n');
    fprintf(fid, '- `time_shift_window_results.csv`\n');
    fprintf(fid, '- `range_shift_window_results.csv`\n');
    fprintf(fid, '- `shift_dataset_build_summary.csv`\n');
    fprintf(fid, '- `shift_sweep_summary.png`\n');
end
