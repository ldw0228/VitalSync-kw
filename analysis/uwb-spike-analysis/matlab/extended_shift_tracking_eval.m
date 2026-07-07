clear; clc; close all;

workspace_dir = 'C:\Users\rkdeh\Documents\Codex\2026-07-01\d-uwb';
data_dir = 'D:\UWB\UWB_BIOPAC_DATA_0601';
out_dir = fullfile(workspace_dir, 'outputs', 'extended_shift_tracking');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fs_uwb_default = 17;
fs_bio_default = 250;
window_sec = 30;
stride_sec = 1;
freq_band_bpm = [6 40];
time_shifts = -10:1:10;
spike_threshold_percentile = 65;

shift_methods = ["bpf_com", "bpf_tv", "bpf_mean"];
function_methods = ["bpf_com", "bpf_tv", "bpf_mean", ...
    "bpf_spike_com_signed", "bpf_spike_tv_signed", ...
    "raw_roi_com", "raw_roi_tv", ...
    "raw_spike_com_signed", "raw_spike_tv_signed"];
estimators = ["fft", "autocorr", "peak_interval"];

shift_pred = cell(numel(shift_methods), numel(time_shifts));
shift_true = cell(numel(shift_methods), numel(time_shifts));
shift_subject = cell(numel(shift_methods), numel(time_shifts));
shift_start = cell(numel(shift_methods), numel(time_shifts));

func_pred = cell(numel(function_methods), numel(estimators));
func_true = cell(numel(function_methods), numel(estimators));
func_subject = cell(numel(function_methods), numel(estimators));
func_start = cell(numel(function_methods), numel(estimators));

zero_series_rows = strings(0, 9);
file_rows = strings(0, 10);

files = dir(fullfile(data_dir, 'SyncData_sub*.mat'));

for fi = 1:numel(files)
    file_path = fullfile(files(fi).folder, files(fi).name);
    file_row = ["", "", "", "", "", "", "", "", "", ""];
    file_row(1) = string(files(fi).name);
    try
        loaded = load(file_path, 'SyncData');
        if ~isfield(loaded, 'SyncData')
            file_row(2) = "skipped";
            file_row(10) = "No SyncData field";
            file_rows(end+1,:) = file_row; %#ok<AGROW>
            continue;
        end

        s = loaded.SyncData;
        if ~has_required_fields(s)
            file_row(2) = "skipped";
            file_row(10) = "Missing required fields";
            if isfield(s, 'subject_num'), file_row(3) = string(double(s.subject_num)); end
            if isfield(s, 'subject_name'), file_row(4) = string(s.subject_name); end
            file_rows(end+1,:) = file_row; %#ok<AGROW>
            continue;
        end

        fs_uwb = get_field_or_default(s, 'Fs_uwb', fs_uwb_default);
        fs_bio = get_field_or_default(s, 'Fs_biopac', fs_bio_default);
        subject_num = double(s.subject_num);
        subject_name = string(s.subject_name);

        [uwb_time, bpf_com, bpf_tv, com_z, tv_z] = prepare_uwb_arrays(s, fs_uwb);
        bio_time = double(s.bio_time_final(:)');
        bio_signal = double(s.bpf_bio_final(:)');
        n_bins = size(com_z, 1);

        d_com = diff(com_z, 1, 2);
        d_tv = diff(tv_z, 1, 2);
        d_time = uwb_time(2:end);
        raw_th_com = prctile(abs(d_com(:)), spike_threshold_percentile);
        raw_th_tv = prctile(abs(d_tv(:)), spike_threshold_percentile);
        bpf_th_com = prctile(abs(diff(bpf_com(:))), spike_threshold_percentile);
        bpf_th_tv = prctile(abs(diff(bpf_tv(:))), spike_threshold_percentile);

        common_start = max(uwb_time(1), bio_time(1));
        common_end = min(uwb_time(end), bio_time(end));
        start_times = common_start:stride_sec:(common_end - window_sec);
        windows_used = 0;
        valid_bio_windows = 0;

        for wi = 1:numel(start_times)
            st_time = start_times(wi);
            ed_time = st_time + window_sec;

            bio_idx = get_time_indices(bio_time, st_time, ed_time);
            if numel(bio_idx) < round(window_sec * fs_bio * 0.8)
                continue;
            end
            rr_bio = estimate_rr_fft(bio_signal(bio_idx), fs_bio, freq_band_bpm);
            if ~isfinite(rr_bio)
                continue;
            end
            valid_bio_windows = valid_bio_windows + 1;

            % Dense one-second time-shift sweep on core BPF signals.
            for si = 1:numel(time_shifts)
                shift_sec = time_shifts(si);
                u_idx = get_time_indices(uwb_time, st_time + shift_sec, ed_time + shift_sec);
                if numel(u_idx) < round(window_sec * fs_uwb * 0.8)
                    continue;
                end
                sigs = build_core_shift_signals(bpf_com(u_idx), bpf_tv(u_idx));
                for mi = 1:numel(shift_methods)
                    pred = estimate_rr_fft(sigs{mi}, fs_uwb, freq_band_bpm);
                    shift_pred{mi, si}(end+1, 1) = pred;
                    shift_true{mi, si}(end+1, 1) = rr_bio;
                    shift_subject{mi, si}(end+1, 1) = subject_num;
                    shift_start{mi, si}(end+1, 1) = st_time;
                end
            end

            % Richer zero-shift function comparison.
            u0_idx = get_time_indices(uwb_time, st_time, ed_time);
            d0_idx = get_time_indices(d_time, st_time, ed_time);
            if numel(u0_idx) >= round(window_sec * fs_uwb * 0.8) && ...
                    numel(d0_idx) >= round(window_sec * fs_uwb * 0.8)
                roi_com = choose_roi(s, "com", st_time, ed_time, d_com(:, d0_idx), n_bins, 20);
                roi_tv = choose_roi(s, "tv", st_time, ed_time, d_tv(:, d0_idx), n_bins, 20);
                sig_map = build_function_signals( ...
                    bpf_com(u0_idx), bpf_tv(u0_idx), ...
                    com_z(roi_com, u0_idx), tv_z(roi_tv, u0_idx), ...
                    d_com(roi_com, d0_idx), d_tv(roi_tv, d0_idx), ...
                    bpf_th_com, bpf_th_tv, raw_th_com, raw_th_tv);

                for mi = 1:numel(function_methods)
                    method = function_methods(mi);
                    sig = sig_map.(char(method));
                    for ei = 1:numel(estimators)
                        estimator = estimators(ei);
                        pred = estimate_rr_by_method(sig, fs_uwb, freq_band_bpm, estimator);
                        func_pred{mi, ei}(end+1, 1) = pred;
                        func_true{mi, ei}(end+1, 1) = rr_bio;
                        func_subject{mi, ei}(end+1, 1) = subject_num;
                        func_start{mi, ei}(end+1, 1) = st_time;
                        if any(method == ["bpf_com", "bpf_tv", "bpf_mean"]) && estimator ~= "peak_interval"
                            zero_series_rows(end+1,:) = [ ...
                                string(files(fi).name), string(subject_num), subject_name, ...
                                string(wi), string(st_time), method, estimator, ...
                                string(rr_bio), string(pred) ...
                            ]; %#ok<AGROW>
                        end
                    end
                end
            end

            windows_used = windows_used + 1;
        end

        file_row = [string(files(fi).name), "used", string(subject_num), subject_name, ...
            string(numel(start_times)), string(valid_bio_windows), string(windows_used), ...
            string(numel(uwb_time)), string(n_bins), "ok"];
        file_rows(end+1,:) = file_row; %#ok<AGROW>
    catch ME
        file_row(2) = "error";
        file_row(10) = string(ME.message);
        file_rows(end+1,:) = file_row; %#ok<AGROW>
    end
end

shift_metrics = summarize_shift_metrics(shift_pred, shift_true, shift_subject, shift_start, shift_methods, time_shifts);
function_metrics = summarize_function_metrics(func_pred, func_true, func_subject, func_start, function_methods, estimators);
best_shift = summarize_best_shift(shift_metrics);

file_table = array2table(file_rows, 'VariableNames', ...
    {'file','status','subject_num','subject_name','candidate_windows','valid_bio_windows', ...
    'windows_used','uwb_frames','range_bins','message'});
zero_series_table = array2table(zero_series_rows, 'VariableNames', ...
    {'file','subject_num','subject_name','window_index','start_sec','method','estimator','rr_bio','rr_uwb'});

writetable(file_table, fullfile(out_dir, 'extended_file_summary.csv'));
writetable(shift_metrics, fullfile(out_dir, 'time_shift_tracking_metrics.csv'));
writetable(function_metrics, fullfile(out_dir, 'zero_shift_function_metrics.csv'));
writetable(best_shift, fullfile(out_dir, 'best_shift_by_method.csv'));
writetable(zero_series_table, fullfile(out_dir, 'zero_shift_tracking_series.csv'));

make_extended_plots(out_dir, shift_metrics, function_metrics, zero_series_table);
write_extended_report(out_dir, shift_metrics, function_metrics, best_shift, file_table, ...
    window_sec, stride_sec, time_shifts, freq_band_bpm, spike_threshold_percentile);

disp(best_shift);
disp(function_metrics(1:min(15, height(function_metrics)), :));
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

function [uwb_time, bpf_com, bpf_tv, com_z, tv_z] = prepare_uwb_arrays(s, fs_uwb)
    uwb_time = double(s.uwb_time(:)');
    bpf_com = double(s.bpf_com(:)');
    bpf_tv = double(s.bpf_tv(:)');
    n_1d = min([numel(uwb_time), numel(bpf_com), numel(bpf_tv)]);
    uwb_time = uwb_time(1:n_1d);
    bpf_com = bpf_com(1:n_1d);
    bpf_tv = bpf_tv(1:n_1d);

    com = double(s.com_final);
    tv = double(s.tv_final);
    n_bins = min(size(com, 1), size(tv, 1));
    n_frames = min([size(com, 2), size(tv, 2), n_1d]);
    com = com(1:n_bins, 1:n_frames);
    tv = tv(1:n_bins, 1:n_frames);
    uwb_time = uwb_time(1:n_frames);
    bpf_com = bpf_com(1:n_frames);
    bpf_tv = bpf_tv(1:n_frames);
    [com_z, tv_z] = robust_subject_zscore(com, tv);

    if isempty(uwb_time) || any(diff(uwb_time) <= 0)
        uwb_time = (0:n_frames-1) / fs_uwb;
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

function sigs = build_core_shift_signals(com, tv)
    com = safe_zscore(com);
    tv = safe_zscore(tv);
    sigs = {com, tv, 0.5 * (com + tv)};
end

function sig_map = build_function_signals(bpf_com, bpf_tv, raw_com_roi, raw_tv_roi, ...
    d_com_roi, d_tv_roi, bpf_th_com, bpf_th_tv, raw_th_com, raw_th_tv)
    bpf_com = safe_zscore(bpf_com);
    bpf_tv = safe_zscore(bpf_tv);
    bpf_mean = 0.5 * (bpf_com + bpf_tv);

    dbpf_com = diff(bpf_com(:)');
    dbpf_tv = diff(bpf_tv(:)');
    bpf_spike_com_signed = double(dbpf_com > bpf_th_com) - double(dbpf_com < -bpf_th_com);
    bpf_spike_tv_signed = double(dbpf_tv > bpf_th_tv) - double(dbpf_tv < -bpf_th_tv);

    raw_roi_com = safe_zscore(mean(raw_com_roi, 1, 'omitnan'));
    raw_roi_tv = safe_zscore(mean(raw_tv_roi, 1, 'omitnan'));
    raw_spike_com_signed = mean(double(d_com_roi > raw_th_com) - double(d_com_roi < -raw_th_com), 1, 'omitnan');
    raw_spike_tv_signed = mean(double(d_tv_roi > raw_th_tv) - double(d_tv_roi < -raw_th_tv), 1, 'omitnan');

    sig_map = struct();
    sig_map.bpf_com = bpf_com;
    sig_map.bpf_tv = bpf_tv;
    sig_map.bpf_mean = bpf_mean;
    sig_map.bpf_spike_com_signed = bpf_spike_com_signed;
    sig_map.bpf_spike_tv_signed = bpf_spike_tv_signed;
    sig_map.raw_roi_com = raw_roi_com;
    sig_map.raw_roi_tv = raw_roi_tv;
    sig_map.raw_spike_com_signed = raw_spike_com_signed;
    sig_map.raw_spike_tv_signed = raw_spike_tv_signed;
end

function x = safe_zscore(x)
    x = double(x(:)');
    x(~isfinite(x)) = 0;
    x = x - mean(x, 'omitnan');
    s = std(x, 0, 'omitnan');
    if ~isfinite(s) || s < 1e-9
        s = 1;
    end
    x = x / s;
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

function rr = estimate_rr_by_method(signal, fs, band_bpm, estimator)
    switch string(estimator)
        case "fft"
            rr = estimate_rr_fft(signal, fs, band_bpm);
        case "autocorr"
            rr = estimate_rr_autocorr(signal, fs, band_bpm);
        case "peak_interval"
            rr = estimate_rr_peak_interval(signal, fs, band_bpm);
        otherwise
            rr = NaN;
    end
end

function rr = estimate_rr_fft(signal, fs, band_bpm)
    signal = clean_signal(signal);
    if numel(signal) < fs * 8 || std(signal) < 1e-9
        rr = NaN;
        return;
    end
    n = numel(signal);
    win = 0.5 - 0.5*cos(2*pi*(0:n-1)'/max(n-1,1));
    signal = signal(:) .* win;
    nfft = 2^nextpow2(max(n, 2048));
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

function rr = estimate_rr_autocorr(signal, fs, band_bpm)
    signal = clean_signal(signal);
    if numel(signal) < fs * 8 || std(signal) < 1e-9
        rr = NaN;
        return;
    end
    min_lag = max(1, floor(fs / (band_bpm(2) / 60)));
    max_lag = min(numel(signal) - 2, ceil(fs / (band_bpm(1) / 60)));
    if max_lag <= min_lag
        rr = NaN;
        return;
    end
    scores = NaN(max_lag - min_lag + 1, 1);
    k = 1;
    for lag = min_lag:max_lag
        a = signal(1:end-lag);
        b = signal(1+lag:end);
        denom = sqrt(sum(a.^2) * sum(b.^2));
        if denom > 1e-12
            scores(k) = sum(a .* b) / denom;
        end
        k = k + 1;
    end
    [best_score, idx] = max(scores);
    if ~isfinite(best_score)
        rr = NaN;
    else
        lag = min_lag + idx - 1;
        rr = 60 * fs / lag;
    end
end

function rr = estimate_rr_peak_interval(signal, fs, band_bpm)
    signal = clean_signal(signal);
    if numel(signal) < fs * 8 || std(signal) < 1e-9
        rr = NaN;
        return;
    end
    min_dist = floor(fs / (band_bpm(2) / 60));
    threshold = median(signal) + 0.2 * std(signal);
    candidates = find(signal(2:end-1) > signal(1:end-2) & signal(2:end-1) >= signal(3:end) & signal(2:end-1) > threshold) + 1;
    if isempty(candidates)
        rr = NaN;
        return;
    end
    peaks = candidates(1);
    for i = 2:numel(candidates)
        if candidates(i) - peaks(end) >= min_dist
            peaks(end+1) = candidates(i); %#ok<AGROW>
        elseif signal(candidates(i)) > signal(peaks(end))
            peaks(end) = candidates(i);
        end
    end
    if numel(peaks) < 3
        rr = NaN;
        return;
    end
    intervals = diff(peaks) / fs;
    bpm = 60 ./ intervals;
    bpm = bpm(bpm >= band_bpm(1) & bpm <= band_bpm(2));
    if isempty(bpm)
        rr = NaN;
    else
        rr = median(bpm);
    end
end

function signal = clean_signal(signal)
    signal = double(signal(:));
    signal = signal(isfinite(signal));
    if isempty(signal)
        return;
    end
    signal = signal - mean(signal);
end

function metrics = summarize_shift_metrics(pred_cells, true_cells, subject_cells, start_cells, methods, shifts)
    rows = strings(0, 19);
    for mi = 1:numel(methods)
        for si = 1:numel(shifts)
            m = compute_metrics(pred_cells{mi, si}, true_cells{mi, si}, subject_cells{mi, si}, start_cells{mi, si});
            rows(end+1,:) = metric_row("time_shift", methods(mi), "fft", shifts(si), m); %#ok<AGROW>
        end
    end
    metrics = metric_table(rows);
end

function metrics = summarize_function_metrics(pred_cells, true_cells, subject_cells, start_cells, methods, estimators)
    rows = strings(0, 19);
    for mi = 1:numel(methods)
        for ei = 1:numel(estimators)
            m = compute_metrics(pred_cells{mi, ei}, true_cells{mi, ei}, subject_cells{mi, ei}, start_cells{mi, ei});
            rows(end+1,:) = metric_row("zero_shift_function", methods(mi), estimators(ei), 0, m); %#ok<AGROW>
        end
    end
    metrics = metric_table(rows);
end

function best = summarize_best_shift(metrics)
    methods = unique(string(metrics.method), 'stable')';
    rows = strings(0, 12);
    for method = methods
        mask = string(metrics.method) == method & string(metrics.source) == "time_shift";
        sub = metrics(mask, :);
        mae = numeric_column(sub.mae_bpm);
        [~, idx] = min(mae);
        zero = sub(numeric_column(sub.shift_sec) == 0, :);
        rows(end+1,:) = [method, string(sub.shift_sec(idx)), string(sub.mae_bpm(idx)), ...
            string(sub.rmse_bpm(idx)), string(sub.corr(idx)), string(sub.trend_acc(idx)), ...
            string(zero.mae_bpm(1)), string(str2double(string(sub.mae_bpm(idx))) - str2double(string(zero.mae_bpm(1)))), ...
            string(sub.within_3bpm(idx)), string(sub.within_5bpm(idx)), ...
            string(sub.best_lag_sec(idx)), string(sub.best_lag_corr(idx))]; %#ok<AGROW>
    end
    best = array2table(rows, 'VariableNames', ...
        {'method','best_shift_sec','best_mae_bpm','best_rmse_bpm','corr','trend_acc', ...
        'zero_shift_mae_bpm','mae_delta_vs_zero','within_3bpm','within_5bpm', ...
        'best_lag_sec','best_lag_corr'});
end

function row = metric_row(source, method, estimator, shift, m)
    row = [source, method, estimator, string(shift), string(m.n), ...
        string(m.mae), string(m.rmse), string(m.median_ae), string(m.p90_ae), ...
        string(m.bias), string(m.sd_error), string(m.within_3), string(m.within_5), ...
        string(m.corr), string(m.r2), string(m.trend_acc), string(m.delta_mae), ...
        string(m.best_lag_sec), string(m.best_lag_corr)];
end

function table_out = metric_table(rows)
    table_out = array2table(rows, 'VariableNames', ...
        {'source','method','estimator','shift_sec','n','mae_bpm','rmse_bpm', ...
        'median_ae_bpm','p90_ae_bpm','bias_bpm','sd_error_bpm','within_3bpm', ...
        'within_5bpm','corr','r2','trend_acc','delta_mae_bpm','best_lag_sec','best_lag_corr'});
end

function m = compute_metrics(pred, target, subject, start_time)
    pred = double(pred(:));
    target = double(target(:));
    subject = double(subject(:));
    start_time = double(start_time(:));
    valid = isfinite(pred) & isfinite(target);
    pred = pred(valid);
    target = target(valid);
    subject = subject(valid);
    start_time = start_time(valid);
    err = pred - target;
    ae = abs(err);

    m = struct('n', numel(pred), 'mae', NaN, 'rmse', NaN, 'median_ae', NaN, ...
        'p90_ae', NaN, 'bias', NaN, 'sd_error', NaN, 'within_3', NaN, ...
        'within_5', NaN, 'corr', NaN, 'r2', NaN, 'trend_acc', NaN, ...
        'delta_mae', NaN, 'best_lag_sec', NaN, 'best_lag_corr', NaN);
    if isempty(pred)
        return;
    end

    m.mae = mean(ae, 'omitnan');
    m.rmse = sqrt(mean(err.^2, 'omitnan'));
    m.median_ae = median(ae, 'omitnan');
    m.p90_ae = percentile_simple(ae, 90);
    m.bias = mean(err, 'omitnan');
    m.sd_error = std(err, 0, 'omitnan');
    m.within_3 = mean(ae <= 3, 'omitnan');
    m.within_5 = mean(ae <= 5, 'omitnan');
    m.corr = simple_corr(pred, target);
    ss_res = sum((target - pred).^2, 'omitnan');
    ss_tot = sum((target - mean(target, 'omitnan')).^2, 'omitnan');
    if ss_tot > 1e-12
        m.r2 = 1 - ss_res / ss_tot;
    end
    [m.trend_acc, m.delta_mae] = tracking_delta_metrics(pred, target, subject, start_time);
    [m.best_lag_sec, m.best_lag_corr] = best_lag_corr(pred, target, subject, start_time, -5:5);
end

function p = percentile_simple(x, q)
    x = sort(x(isfinite(x)));
    if isempty(x)
        p = NaN;
        return;
    end
    idx = max(1, min(numel(x), round((q / 100) * numel(x))));
    p = x(idx);
end

function [trend_acc, delta_mae] = tracking_delta_metrics(pred, target, subject, start_time)
    trend_hits = [];
    delta_errors = [];
    subjects = unique(subject(:))';
    for s = subjects
        mask = subject == s;
        [~, order] = sort(start_time(mask));
        p = pred(mask);
        t = target(mask);
        p = p(order);
        t = t(order);
        if numel(p) < 2
            continue;
        end
        dp = diff(p);
        dt = diff(t);
        use = abs(dt) >= 0.5;
        if any(use)
            trend_hits = [trend_hits; sign(dp(use)) == sign(dt(use))]; %#ok<AGROW>
        end
        delta_errors = [delta_errors; abs(dp - dt)]; %#ok<AGROW>
    end
    trend_acc = mean(double(trend_hits), 'omitnan');
    delta_mae = mean(delta_errors, 'omitnan');
end

function [best_lag_sec, best_corr] = best_lag_corr(pred, target, subject, start_time, lags)
    best_lag_sec = NaN;
    best_corr = NaN;
    corr_values = NaN(size(lags));
    for li = 1:numel(lags)
        lag = lags(li);
        p_all = [];
        t_all = [];
        subjects = unique(subject(:))';
        for s = subjects
            mask = subject == s;
            [~, order] = sort(start_time(mask));
            p = pred(mask);
            t = target(mask);
            p = p(order);
            t = t(order);
            if numel(p) <= abs(lag) + 2
                continue;
            end
            if lag > 0
                p_all = [p_all; p(1+lag:end)]; %#ok<AGROW>
                t_all = [t_all; t(1:end-lag)]; %#ok<AGROW>
            elseif lag < 0
                k = abs(lag);
                p_all = [p_all; p(1:end-k)]; %#ok<AGROW>
                t_all = [t_all; t(1+k:end)]; %#ok<AGROW>
            else
                p_all = [p_all; p]; %#ok<AGROW>
                t_all = [t_all; t]; %#ok<AGROW>
            end
        end
        corr_values(li) = simple_corr(p_all, t_all);
    end
    if any(isfinite(corr_values))
        [best_corr, idx] = max(corr_values);
        best_lag_sec = lags(idx);
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

function values = numeric_column(col)
    if isnumeric(col)
        values = double(col);
    else
        values = str2double(string(col));
    end
end

function make_extended_plots(out_dir, shift_metrics, function_metrics, zero_series)
    fig = figure('Visible', 'off', 'Position', [100 100 1050 700]);
    hold on;
    methods = unique(string(shift_metrics.method), 'stable')';
    for method = methods
        mask = string(shift_metrics.method) == method;
        shifts = numeric_column(shift_metrics.shift_sec(mask));
        mae = numeric_column(shift_metrics.mae_bpm(mask));
        plot(shifts, mae, '-o', 'LineWidth', 1.5, 'DisplayName', method);
    end
    xlabel('UWB time shift relative to BIOPAC window (sec)');
    ylabel('RR MAE (bpm)');
    title('One-second time shift sweep');
    grid on;
    legend('Location', 'best');
    exportgraphics(fig, fullfile(out_dir, 'time_shift_mae_curve.png'), 'Resolution', 160);
    close(fig);

    fig = figure('Visible', 'off', 'Position', [100 100 1200 750]);
    methods = string(function_metrics.method);
    estimators = string(function_metrics.estimator);
    labels = methods + " / " + estimators;
    mae = numeric_column(function_metrics.mae_bpm);
    [mae_sorted, order] = sort(mae);
    top_n = min(15, numel(order));
    bar(mae_sorted(1:top_n));
    xticklabels(labels(order(1:top_n)));
    xtickangle(35);
    ylabel('RR MAE (bpm)');
    title('Best zero-shift function candidates');
    grid on;
    exportgraphics(fig, fullfile(out_dir, 'zero_shift_function_mae_top15.png'), 'Resolution', 160);
    close(fig);

    if height(zero_series) > 0
        make_tracking_example(out_dir, zero_series);
    end
end

function make_tracking_example(out_dir, zero_series)
    subject_col = numeric_column(zero_series.subject_num);
    subjects = unique(subject_col);
    subject = subjects(1);
    mask_subject = subject_col == subject;
    start_sec = numeric_column(zero_series.start_sec);
    rr_bio = numeric_column(zero_series.rr_bio);
    method = string(zero_series.method);
    estimator = string(zero_series.estimator);

    fig = figure('Visible', 'off', 'Position', [100 100 1200 650]);
    hold on;
    base_mask = mask_subject & method == "bpf_com" & estimator == "fft";
    [x, order] = sort(start_sec(base_mask));
    y = rr_bio(base_mask);
    plot(x, y(order), 'k-', 'LineWidth', 2, 'DisplayName', 'BIOPAC');

    candidates = [
        "bpf_com", "fft";
        "bpf_tv", "fft";
        "bpf_mean", "fft";
        "bpf_mean", "autocorr"
    ];
    for i = 1:size(candidates, 1)
        mask = mask_subject & method == candidates(i, 1) & estimator == candidates(i, 2);
        [x, order] = sort(start_sec(mask));
        y = numeric_column(zero_series.rr_uwb(mask));
        plot(x, y(order), 'LineWidth', 1.2, 'DisplayName', candidates(i,1) + "/" + candidates(i,2));
    end
    xlabel('Window start time (sec)');
    ylabel('RR (bpm)');
    title(sprintf('Tracking example subject %g, 30s window / 1s stride', subject));
    grid on;
    legend('Location', 'best');
    exportgraphics(fig, fullfile(out_dir, 'tracking_example_subject_first.png'), 'Resolution', 160);
    close(fig);
end

function write_extended_report(out_dir, shift_metrics, function_metrics, best_shift, file_table, ...
    window_sec, stride_sec, time_shifts, freq_band_bpm, spike_threshold)
    report_path = fullfile(out_dir, 'extended_shift_tracking_summary.md');
    fid = fopen(report_path, 'w', 'n', 'UTF-8');
    cleanup = onCleanup(@() fclose(fid));

    func_mae = numeric_column(function_metrics.mae_bpm);
    [~, best_func_idx] = min(func_mae);
    used = sum(string(file_table.status) == "used");
    skipped = sum(string(file_table.status) ~= "used");
    total_windows = sum(numeric_column(file_table.windows_used), 'omitnan');

    fprintf(fid, '# Extended Shift Tracking Evaluation\n\n');
    fprintf(fid, '## Setup\n\n');
    fprintf(fid, '- Window: %d sec\n', window_sec);
    fprintf(fid, '- Window stride: %d sec\n', stride_sec);
    fprintf(fid, '- Time shift sweep: %g to %g sec, 1 sec step\n', min(time_shifts), max(time_shifts));
    fprintf(fid, '- BIOPAC reference RR: FFT dominant frequency in %.0f-%.0f bpm band\n', freq_band_bpm(1), freq_band_bpm(2));
    fprintf(fid, '- Spike threshold percentile: %.0f\n', spike_threshold);
    fprintf(fid, '- Used files: %d\n', used);
    fprintf(fid, '- Skipped/error files: %d\n', skipped);
    fprintf(fid, '- Tracking windows used: %.0f\n\n', total_windows);

    fprintf(fid, '## Best Time Shift\n\n');
    fprintf(fid, '| Method | Best shift sec | Best MAE | Zero-shift MAE | Delta | Corr | Trend acc | Within 5 bpm |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|\n');
    for i = 1:height(best_shift)
        fprintf(fid, '| %s | %s | %.2f | %.2f | %.2f | %.3f | %.3f | %.3f |\n', ...
            string(best_shift.method(i)), string(best_shift.best_shift_sec(i)), ...
            str2double(string(best_shift.best_mae_bpm(i))), ...
            str2double(string(best_shift.zero_shift_mae_bpm(i))), ...
            str2double(string(best_shift.mae_delta_vs_zero(i))), ...
            str2double(string(best_shift.corr(i))), ...
            str2double(string(best_shift.trend_acc(i))), ...
            str2double(string(best_shift.within_5bpm(i))));
    end

    fprintf(fid, '\n## Best Zero-Shift Function\n\n');
    fprintf(fid, 'Best candidate: `%s / %s`, MAE %.2f bpm, RMSE %.2f bpm, corr %.3f, trend acc %.3f.\n\n', ...
        string(function_metrics.method(best_func_idx)), string(function_metrics.estimator(best_func_idx)), ...
        str2double(string(function_metrics.mae_bpm(best_func_idx))), ...
        str2double(string(function_metrics.rmse_bpm(best_func_idx))), ...
        str2double(string(function_metrics.corr(best_func_idx))), ...
        str2double(string(function_metrics.trend_acc(best_func_idx))));

    fprintf(fid, 'Top zero-shift candidates by MAE:\n\n');
    fprintf(fid, '| Method | Estimator | MAE | RMSE | Median AE | P90 AE | Corr | Trend acc | Delta MAE |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---:|---:|---:|---:|\n');
    [~, order] = sort(func_mae);
    for k = 1:min(12, numel(order))
        i = order(k);
        fprintf(fid, '| %s | %s | %.2f | %.2f | %.2f | %.2f | %.3f | %.3f | %.2f |\n', ...
            string(function_metrics.method(i)), string(function_metrics.estimator(i)), ...
            str2double(string(function_metrics.mae_bpm(i))), ...
            str2double(string(function_metrics.rmse_bpm(i))), ...
            str2double(string(function_metrics.median_ae_bpm(i))), ...
            str2double(string(function_metrics.p90_ae_bpm(i))), ...
            str2double(string(function_metrics.corr(i))), ...
            str2double(string(function_metrics.trend_acc(i))), ...
            str2double(string(function_metrics.delta_mae_bpm(i))));
    end

    fprintf(fid, '\n## Interpretation Notes\n\n');
    fprintf(fid, '- MAE/RMSE measure point-wise RR error against BIOPAC.\n');
    fprintf(fid, '- Trend acc checks whether UWB follows the direction of BIOPAC RR change between adjacent 1-sec-shifted windows. Changes smaller than 0.5 bpm are ignored.\n');
    fprintf(fid, '- Delta MAE measures how close the RR change amount is, not only the absolute RR value.\n');
    fprintf(fid, '- Best lag is computed over -5 to +5 tracking windows. Because stride is 1 sec, this is also seconds.\n');
    fprintf(fid, '- If best time shift is far from zero but improves MAE only slightly, sync is probably not the main bottleneck.\n\n');

    fprintf(fid, '## Output Files\n\n');
    fprintf(fid, '- `time_shift_tracking_metrics.csv`\n');
    fprintf(fid, '- `zero_shift_function_metrics.csv`\n');
    fprintf(fid, '- `best_shift_by_method.csv`\n');
    fprintf(fid, '- `zero_shift_tracking_series.csv`\n');
    fprintf(fid, '- `extended_file_summary.csv`\n');
    fprintf(fid, '- `time_shift_mae_curve.png`\n');
    fprintf(fid, '- `zero_shift_function_mae_top15.png`\n');
    fprintf(fid, '- `tracking_example_subject_first.png`\n');
end
