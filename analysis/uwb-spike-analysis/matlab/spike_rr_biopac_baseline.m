clear; clc; close all;

workspace_dir = 'C:\Users\rkdeh\Documents\Codex\2026-07-01\d-uwb';
data_dir = 'D:\UWB\UWB_BIOPAC_DATA_0601';
out_dir = fullfile(workspace_dir, 'outputs', 'spike_rr_biopac');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fs_uwb_default = 17;
fs_bio_default = 250;
window_sec = 30;
stride_sec = 15;
freq_band_bpm = [6 40];
threshold_percentiles = [55 65 75 85];
roi_half_width = 20;

files = dir(fullfile(data_dir, 'SyncData_sub*.mat'));
all_rows = strings(0, 24);
summary_rows = strings(0, 8);

for fi = 1:numel(files)
    file_path = fullfile(files(fi).folder, files(fi).name);
    row_summary = ["", "", "", "", "", "", "", ""];
    row_summary(1) = string(files(fi).name);
    try
        loaded = load(file_path, 'SyncData');
        if ~isfield(loaded, 'SyncData')
            row_summary(2) = "skipped";
            row_summary(8) = "No SyncData field";
            summary_rows(end+1,:) = row_summary; %#ok<AGROW>
            continue;
        end
        s = loaded.SyncData;
        if ~has_required_fields(s)
            row_summary(2) = "skipped";
            row_summary(8) = "Missing final UWB/BIOPAC fields";
            if isfield(s, 'subject_num'), row_summary(3) = string(double(s.subject_num)); end
            if isfield(s, 'subject_name'), row_summary(4) = string(s.subject_name); end
            summary_rows(end+1,:) = row_summary; %#ok<AGROW>
            continue;
        end

        fs_uwb = get_field_or_default(s, 'Fs_uwb', fs_uwb_default);
        fs_bio = get_field_or_default(s, 'Fs_biopac', fs_bio_default);
        subject_num = double(s.subject_num);
        subject_name = string(s.subject_name);

        com = double(s.com_final);
        tv = double(s.tv_final);
        n_bins = min(size(com, 1), size(tv, 1));
        n_frames = min(size(com, 2), size(tv, 2));
        com = com(1:n_bins, 1:n_frames);
        tv = tv(1:n_bins, 1:n_frames);

        if isfield(s, 'time_final')
            uwb_time = double(s.time_final(:)');
            uwb_time = uwb_time(1:min(numel(uwb_time), n_frames));
        else
            uwb_time = (0:n_frames-1) / fs_uwb;
        end
        n_frames = min(n_frames, numel(uwb_time));
        com = com(:, 1:n_frames);
        tv = tv(:, 1:n_frames);

        [com_z, tv_z] = robust_subject_zscore(com, tv);
        d_com = diff(com_z, 1, 2);
        d_tv = diff(tv_z, 1, 2);
        d_time = uwb_time(2:end);

        threshold_values = zeros(numel(threshold_percentiles), 2);
        bpf_threshold_values = NaN(numel(threshold_percentiles), 2);
        for ti = 1:numel(threshold_percentiles)
            threshold_values(ti, 1) = prctile(abs(d_com(:)), threshold_percentiles(ti));
            threshold_values(ti, 2) = prctile(abs(d_tv(:)), threshold_percentiles(ti));
            if isfield(s, 'bpf_com') && isfield(s, 'bpf_tv')
                dbpf_com_all = diff(double(s.bpf_com(:)));
                dbpf_tv_all = diff(double(s.bpf_tv(:)));
                bpf_threshold_values(ti, 1) = prctile(abs(dbpf_com_all(isfinite(dbpf_com_all))), threshold_percentiles(ti));
                bpf_threshold_values(ti, 2) = prctile(abs(dbpf_tv_all(isfinite(dbpf_tv_all))), threshold_percentiles(ti));
            end
        end

        windows_used = 0;
        start_times = uwb_time(1):stride_sec:(uwb_time(end) - window_sec);
        for wi = 1:numel(start_times)
            st_time = start_times(wi);
            ed_time = st_time + window_sec;

            bio_idx = get_time_indices(s.bio_time_final, st_time, ed_time);
            if numel(bio_idx) < round(window_sec * fs_bio * 0.8)
                continue;
            end
            bio_sig = double(s.bpf_bio_final(bio_idx));
            rr_bio = estimate_rr_fft(bio_sig, fs_bio, freq_band_bpm);
            if ~isfinite(rr_bio)
                continue;
            end

            d_idx = find(d_time >= st_time & d_time < ed_time);
            if numel(d_idx) < round(window_sec * fs_uwb * 0.8)
                continue;
            end

            com_roi = choose_roi(s, "com", st_time, ed_time, d_com(:, d_idx), n_bins, roi_half_width);
            tv_roi = choose_roi(s, "tv", st_time, ed_time, d_tv(:, d_idx), n_bins, roi_half_width);

            bpf_com_rr = NaN;
            bpf_tv_rr = NaN;
            bpf_com_seg = [];
            bpf_tv_seg = [];
            if isfield(s, 'bpf_com') && isfield(s, 'uwb_time')
                bpf_idx = get_time_indices(s.uwb_time, st_time, ed_time);
                if numel(bpf_idx) >= round(window_sec * fs_uwb * 0.8)
                    bpf_com_seg = double(s.bpf_com(bpf_idx));
                    bpf_tv_seg = double(s.bpf_tv(bpf_idx));
                    bpf_com_rr = estimate_rr_fft(bpf_com_seg, fs_uwb, freq_band_bpm);
                    bpf_tv_rr = estimate_rr_fft(bpf_tv_seg, fs_uwb, freq_band_bpm);
                end
            end

            for ti = 1:numel(threshold_percentiles)
                th_com = threshold_values(ti, 1);
                th_tv = threshold_values(ti, 2);

                seg_com = d_com(com_roi, d_idx);
                seg_tv = d_tv(tv_roi, d_idx);
                spike_com_signed = mean(double(seg_com > th_com) - double(seg_com < -th_com), 1);
                spike_tv_signed = mean(double(seg_tv > th_tv) - double(seg_tv < -th_tv), 1);
                spike_com_count = mean(double(abs(seg_com) > th_com), 1);
                spike_tv_count = mean(double(abs(seg_tv) > th_tv), 1);

                rr_spike_com_signed = estimate_rr_fft(spike_com_signed, fs_uwb, freq_band_bpm);
                rr_spike_tv_signed = estimate_rr_fft(spike_tv_signed, fs_uwb, freq_band_bpm);
                rr_spike_com_count = estimate_rr_fft(spike_com_count, fs_uwb, freq_band_bpm);
                rr_spike_tv_count = estimate_rr_fft(spike_tv_count, fs_uwb, freq_band_bpm);
                rr_spike_bpf_com_signed = NaN;
                rr_spike_bpf_tv_signed = NaN;
                rr_spike_bpf_com_count = NaN;
                rr_spike_bpf_tv_count = NaN;
                if ~isempty(bpf_com_seg) && isfinite(bpf_threshold_values(ti, 1))
                    dbpf_com = diff(bpf_com_seg(:))';
                    dbpf_tv = diff(bpf_tv_seg(:))';
                    bpf_com_signed = double(dbpf_com > bpf_threshold_values(ti, 1)) - double(dbpf_com < -bpf_threshold_values(ti, 1));
                    bpf_tv_signed = double(dbpf_tv > bpf_threshold_values(ti, 2)) - double(dbpf_tv < -bpf_threshold_values(ti, 2));
                    bpf_com_count = double(abs(dbpf_com) > bpf_threshold_values(ti, 1));
                    bpf_tv_count = double(abs(dbpf_tv) > bpf_threshold_values(ti, 2));
                    rr_spike_bpf_com_signed = estimate_rr_fft(bpf_com_signed, fs_uwb, freq_band_bpm);
                    rr_spike_bpf_tv_signed = estimate_rr_fft(bpf_tv_signed, fs_uwb, freq_band_bpm);
                    rr_spike_bpf_com_count = estimate_rr_fft(bpf_com_count, fs_uwb, freq_band_bpm);
                    rr_spike_bpf_tv_count = estimate_rr_fft(bpf_tv_count, fs_uwb, freq_band_bpm);
                end

                movement_score = mean([mean(spike_com_count), mean(spike_tv_count)]);

                all_rows(end+1,:) = [ ...
                    string(files(fi).name), string(subject_num), subject_name, ...
                    string(wi), string(st_time), string(ed_time), string(threshold_percentiles(ti)), ...
                    string(rr_bio), string(bpf_com_rr), string(bpf_tv_rr), ...
                    string(rr_spike_com_signed), string(rr_spike_tv_signed), ...
                    string(rr_spike_com_count), string(rr_spike_tv_count), ...
                    string(rr_spike_bpf_com_signed), string(rr_spike_bpf_tv_signed), ...
                    string(rr_spike_bpf_com_count), string(rr_spike_bpf_tv_count), ...
                    string(abs(rr_spike_com_signed - rr_bio)), ...
                    string(abs(rr_spike_tv_signed - rr_bio)), ...
                    string(abs(rr_spike_bpf_com_signed - rr_bio)), ...
                    string(abs(rr_spike_bpf_tv_signed - rr_bio)), ...
                    string(abs(bpf_com_rr - rr_bio)), string(movement_score) ...
                ]; %#ok<AGROW>
            end
            windows_used = windows_used + 1;
        end

        row_summary = [string(files(fi).name), "used", string(subject_num), subject_name, ...
            string(windows_used), string(n_frames), string(n_bins), "ok"];
        summary_rows(end+1,:) = row_summary; %#ok<AGROW>
    catch ME
        row_summary(2) = "error";
        row_summary(8) = string(ME.message);
        summary_rows(end+1,:) = row_summary; %#ok<AGROW>
    end
end

headers = {'file','subject_num','subject_name','window_index','start_sec','end_sec','threshold_percentile', ...
    'rr_bio','rr_bpf_com','rr_bpf_tv','rr_spike_com_signed','rr_spike_tv_signed', ...
    'rr_spike_com_count','rr_spike_tv_count','rr_spike_bpf_com_signed','rr_spike_bpf_tv_signed', ...
    'rr_spike_bpf_com_count','rr_spike_bpf_tv_count','ae_spike_com_signed','ae_spike_tv_signed', ...
    'ae_spike_bpf_com_signed','ae_spike_bpf_tv_signed', ...
    'ae_bpf_com','movement_score'};
window_table = array2table(all_rows, 'VariableNames', headers);
writetable(window_table, fullfile(out_dir, 'rr_window_results.csv'));

summary_headers = {'file','status','subject_num','subject_name','windows_used','frames','bins','message'};
build_summary = array2table(summary_rows, 'VariableNames', summary_headers);
writetable(build_summary, fullfile(out_dir, 'rr_dataset_build_summary.csv'));

metrics_table = summarize_metrics(window_table, threshold_percentiles);
writetable(metrics_table, fullfile(out_dir, 'rr_metrics_summary.csv'));

make_rr_plots(out_dir, window_table, metrics_table);
write_rr_report(out_dir, window_table, metrics_table, build_summary, window_sec, stride_sec, freq_band_bpm);

disp(metrics_table);
fprintf('Outputs written to %s\n', out_dir);

function ok = has_required_fields(s)
    ok = isfield(s, 'com_final') && isfield(s, 'tv_final') && ...
        isfield(s, 'bpf_bio_final') && isfield(s, 'bio_time_final') && ...
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

function roi = choose_roi(s, kind, st_time, ed_time, seg_d, n_bins, half_width)
    peak_field = "snr_peak_bin_" + kind;
    centers_field = "snr_window_centers";
    if isfield(s, peak_field) && isfield(s, centers_field)
        centers = double(s.(centers_field)(:));
        peaks = double(s.(peak_field)(:));
        mask = centers >= st_time & centers < ed_time & isfinite(peaks);
        if any(mask)
            peak_bin = round(median(peaks(mask), 'omitnan'));
        else
            [~, peak_bin] = max(mean(abs(seg_d), 2, 'omitnan'));
        end
    else
        [~, peak_bin] = max(mean(abs(seg_d), 2, 'omitnan'));
    end
    peak_bin = min(max(peak_bin, 1), n_bins);
    roi = max(1, peak_bin-half_width):min(n_bins, peak_bin+half_width);
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

function metrics_table = summarize_metrics(window_table, thresholds)
    metric_rows = strings(0, 8);
    measures = {
        'rr_bpf_com', 'BPF COM';
        'rr_bpf_tv', 'BPF TV';
        'rr_spike_com_signed', 'Spike COM signed';
        'rr_spike_tv_signed', 'Spike TV signed';
        'rr_spike_com_count', 'Spike COM count';
        'rr_spike_tv_count', 'Spike TV count';
        'rr_spike_bpf_com_signed', 'BPF spike COM signed';
        'rr_spike_bpf_tv_signed', 'BPF spike TV signed';
        'rr_spike_bpf_com_count', 'BPF spike COM count';
        'rr_spike_bpf_tv_count', 'BPF spike TV count'
    };
    rr_bio = numeric_column(window_table.rr_bio);
    movement = numeric_column(window_table.movement_score);
    threshold_col = numeric_column(window_table.threshold_percentile);
    for ti = 1:numel(thresholds)
        th_mask = threshold_col == thresholds(ti);
        movement_cut = median(movement(th_mask), 'omitnan');
        for mi = 1:size(measures, 1)
            pred = numeric_column(window_table.(measures{mi,1}));
            err = abs(pred - rr_bio);
            valid = th_mask & isfinite(err);
            stable = valid & movement <= movement_cut;
            moving = valid & movement > movement_cut;
            metric_rows(end+1,:) = [string(measures{mi,2}), string(thresholds(ti)), ...
                string(sum(valid)), string(mean(err(valid), 'omitnan')), ...
                string(median(err(valid), 'omitnan')), string(mean(err(stable), 'omitnan')), ...
                string(mean(err(moving), 'omitnan')), string(simple_corr(pred(valid), rr_bio(valid)))]; %#ok<AGROW>
        end
    end
    metrics_table = array2table(metric_rows, 'VariableNames', ...
        {'method','threshold_percentile','n','mae_bpm','median_ae_bpm','stable_half_mae_bpm','moving_half_mae_bpm','corr_with_bio'});
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

function values = numeric_column(col)
    if isnumeric(col)
        values = double(col);
    else
        values = str2double(string(col));
    end
end

function make_rr_plots(out_dir, window_table, metrics_table)
    methods = string(metrics_table.method);
    thresholds = numeric_column(metrics_table.threshold_percentile);
    mae = numeric_column(metrics_table.mae_bpm);
    spike_mask = contains(methods, "Spike");
    bpf_mask = contains(methods, "BPF");

    fig = figure('Visible', 'off', 'Position', [100 100 1200 700]);
    tiledlayout(1, 2, 'TileSpacing', 'compact');
    nexttile; hold on;
    spike_methods = unique(methods(spike_mask), 'stable')';
    for m = spike_methods
        mask = methods == m;
        plot(thresholds(mask), mae(mask), '-o', 'LineWidth', 1.5, 'DisplayName', m);
    end
    xlabel('Delta threshold percentile');
    ylabel('MAE vs BIOPAC RR (bpm)');
    title('Spike RR threshold sweep');
    grid on; legend('Location', 'best');

    nexttile;
    best_rows = strings(0, 2);
    method_list = unique(methods, 'stable')';
    for m = method_list
        mask = methods == m;
        best_rows(end+1,:) = [m, string(min(mae(mask)))]; %#ok<AGROW>
    end
    bar(str2double(best_rows(:,2)));
    xticklabels(best_rows(:,1));
    xtickangle(35);
    ylabel('Best MAE (bpm)');
    title('Best window RR MAE by method');
    grid on;
    exportgraphics(fig, fullfile(out_dir, 'rr_mae_summary.png'), 'Resolution', 160);
    close(fig);

    make_scatter(window_table, 'rr_bpf_com', NaN, ...
        fullfile(out_dir, 'rr_biopac_vs_bpf_com_scatter.png'), 'BIOPAC vs BPF COM RR');
    make_best_method_scatter(window_table, metrics_table, "BPF spike", ...
        fullfile(out_dir, 'rr_biopac_vs_bpf_spike_scatter.png'), 'BIOPAC vs best BPF-spike RR');
    make_best_method_scatter(window_table, metrics_table, "Spike ", ...
        fullfile(out_dir, 'rr_biopac_vs_range_spike_scatter.png'), 'BIOPAC vs best range-spike RR');
end

function make_best_method_scatter(window_table, metrics_table, method_prefix, out_path, title_text)
    methods = string(metrics_table.method);
    maes = numeric_column(metrics_table.mae_bpm);
    candidate = startsWith(methods, method_prefix);
    [~, local_idx] = min(maes(candidate));
    candidate_indices = find(candidate);
    best_idx = candidate_indices(local_idx);
    method_name = methods(best_idx);
    threshold = str2double(string(metrics_table.threshold_percentile(best_idx)));
    column = method_to_column(method_name);
    make_scatter(window_table, column, threshold, out_path, sprintf('%s: %s, thr %.0f', title_text, method_name, threshold));
end

function column = method_to_column(method_name)
    switch string(method_name)
        case "Spike COM signed"
            column = 'rr_spike_com_signed';
        case "Spike TV signed"
            column = 'rr_spike_tv_signed';
        case "Spike COM count"
            column = 'rr_spike_com_count';
        case "Spike TV count"
            column = 'rr_spike_tv_count';
        case "BPF spike COM signed"
            column = 'rr_spike_bpf_com_signed';
        case "BPF spike TV signed"
            column = 'rr_spike_bpf_tv_signed';
        case "BPF spike COM count"
            column = 'rr_spike_bpf_com_count';
        case "BPF spike TV count"
            column = 'rr_spike_bpf_tv_count';
        otherwise
            column = 'rr_bpf_com';
    end
end

function make_scatter(window_table, pred_column, threshold, out_path, title_text)
    rr_bio = numeric_column(window_table.rr_bio);
    pred = numeric_column(window_table.(pred_column));
    threshold_col = numeric_column(window_table.threshold_percentile);
    if isfinite(threshold)
        mask = threshold_col == threshold & isfinite(rr_bio) & isfinite(pred);
    else
        mask = isfinite(rr_bio) & isfinite(pred);
    end
    fig = figure('Visible', 'off', 'Position', [100 100 800 750]);
    scatter(rr_bio(mask), pred(mask), 18, 'filled', 'MarkerFaceAlpha', 0.35);
    hold on;
    lims = [min([rr_bio(mask); pred(mask)]), max([rr_bio(mask); pred(mask)])];
    plot(lims, lims, 'k--', 'LineWidth', 1.2);
    xlabel('BIOPAC RR (bpm)');
    ylabel(strrep(pred_column, '_', '\_'));
    title(title_text, 'Interpreter', 'none');
    grid on;
    exportgraphics(fig, out_path, 'Resolution', 160);
    close(fig);
end

function write_rr_report(out_dir, window_table, metrics_table, build_summary, window_sec, stride_sec, freq_band_bpm)
    report_path = fullfile(out_dir, 'spike_rr_biopac_summary_ko.md');
    fid = fopen(report_path, 'w', 'n', 'UTF-8');
    cleanup = onCleanup(@() fclose(fid));

    mae = numeric_column(metrics_table.mae_bpm);
    [best_mae, best_idx] = min(mae);
    best_method = string(metrics_table.method(best_idx));
    best_threshold = string(metrics_table.threshold_percentile(best_idx));
    method_names = string(metrics_table.method);
    bpf_spike_mask = startsWith(method_names, "BPF spike");
    range_spike_mask = startsWith(method_names, "Spike ");
    [best_bpf_spike_mae, bpf_local] = min(mae(bpf_spike_mask));
    bpf_indices = find(bpf_spike_mask);
    best_bpf_spike_idx = bpf_indices(bpf_local);
    [best_range_spike_mae, range_local] = min(mae(range_spike_mask));
    range_indices = find(range_spike_mask);
    best_range_spike_idx = range_indices(range_local);

    used = sum(string(build_summary.status) == "used");
    skipped = sum(string(build_summary.status) ~= "used");
    n_windows = height(unique(window_table(:, {'file','window_index'})));

    fprintf(fid, '# Spike RR vs BIOPAC 비교 요약\n\n');
    fprintf(fid, '## 한 줄 결론\n\n');
    fprintf(fid, '이번 실험은 BIOPAC 호흡수와 UWB spike 기반 호흡수 추정을 직접 비교했다. 전체 best는 `%s`, threshold `%s`이며 MAE는 %.2f bpm이다.\n\n', ...
        best_method, best_threshold, best_mae);
    fprintf(fid, 'Spike 계열만 보면 best BPF-spike는 `%s` threshold `%s` MAE %.2f bpm이고, raw range-bin spike best는 `%s` threshold `%s` MAE %.2f bpm이다.\n\n', ...
        string(metrics_table.method(best_bpf_spike_idx)), string(metrics_table.threshold_percentile(best_bpf_spike_idx)), best_bpf_spike_mae, ...
        string(metrics_table.method(best_range_spike_idx)), string(metrics_table.threshold_percentile(best_range_spike_idx)), best_range_spike_mae);

    fprintf(fid, '## 설정\n\n');
    fprintf(fid, '- Window: %d초\n', window_sec);
    fprintf(fid, '- Stride: %d초\n', stride_sec);
    fprintf(fid, '- RR 탐색 대역: %.0f-%.0f bpm\n', freq_band_bpm(1), freq_band_bpm(2));
    fprintf(fid, '- 사용 파일: %d개\n', used);
    fprintf(fid, '- 제외/오류 파일: %d개\n', skipped);
    fprintf(fid, '- BIOPAC 비교 window: %d개\n\n', n_windows);

    fprintf(fid, '## 방법별 성능\n\n');
    fprintf(fid, '| Method | Threshold | N | MAE bpm | Median AE | Stable half MAE | Moving half MAE | Corr |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|\n');
    for i = 1:height(metrics_table)
        fprintf(fid, '| %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %.3f |\n', ...
            string(metrics_table.method(i)), string(metrics_table.threshold_percentile(i)), ...
            string(metrics_table.n(i)), str2double(string(metrics_table.mae_bpm(i))), ...
            str2double(string(metrics_table.median_ae_bpm(i))), ...
            str2double(string(metrics_table.stable_half_mae_bpm(i))), ...
            str2double(string(metrics_table.moving_half_mae_bpm(i))), ...
            str2double(string(metrics_table.corr_with_bio(i))));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## 해석\n\n');
    fprintf(fid, '- 이 실험은 이전 0-13 라벨 분류와 달리 BIOPAC 기준 RR을 직접 타깃으로 삼았다.\n');
    fprintf(fid, '- 이동이 섞인 데이터라 window별 dominant frequency가 흔들릴 수 있고, spike count 계열은 움직임 이벤트에 민감하다.\n');
    fprintf(fid, '- `Stable half`와 `Moving half` 차이가 크면 movement rejection 또는 stable detector가 다음 핵심 개선 포인트다.\n');
    fprintf(fid, '- 기존 `bpf_com/tv`와 spike 계열을 같이 비교했으므로, spike가 단순 전처리보다 나은지/못한지 바로 볼 수 있다.\n\n');

    fprintf(fid, '## 생성 파일\n\n');
    fprintf(fid, '- `rr_window_results.csv`\n');
    fprintf(fid, '- `rr_metrics_summary.csv`\n');
    fprintf(fid, '- `rr_dataset_build_summary.csv`\n');
    fprintf(fid, '- `rr_mae_summary.png`\n');
    fprintf(fid, '- `rr_biopac_vs_bpf_com_scatter.png`\n');
    fprintf(fid, '- `rr_biopac_vs_bpf_spike_scatter.png`\n');
    fprintf(fid, '- `rr_biopac_vs_range_spike_scatter.png`\n');
end
