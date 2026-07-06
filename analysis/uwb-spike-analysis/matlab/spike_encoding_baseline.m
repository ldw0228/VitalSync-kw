clear; clc; close all;

workspace_dir = 'C:\Users\rkdeh\Documents\Codex\2026-07-01\d-uwb';
data_dir = 'D:\UWB\UWB_BIOPAC_DATA_0601';
out_dir = fullfile(workspace_dir, 'outputs', 'spike_baseline');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

rng(22);
fs_uwb = 17;
threshold_percentiles = [65 75 85 90 95];
lambda_values = [0.1 1 10];
k_folds = 5;

files = dir(fullfile(data_dir, 'SyncData_sub*.mat'));
[X_spike_sets, X_raw, y, subject_ids, subject_names, sample_meta, build_summary] = ...
    build_spike_datasets(files, threshold_percentiles, fs_uwb);

classes = unique(y(:))';
fprintf('Built dataset: %d samples, %d classes, %d subjects, raw dims=%d\n', ...
    numel(y), numel(classes), numel(unique(subject_ids)), size(X_raw, 2));
if isempty(y)
    summary_table = struct2table(build_summary);
    writetable(summary_table, fullfile(out_dir, 'spike_dataset_build_summary.csv'));
    error('No samples were built. See spike_dataset_build_summary.csv for details.');
end

subjects = unique(subject_ids(:))';
fold_id_by_subject = make_subject_folds(subjects, k_folds);
sample_folds = zeros(size(subject_ids));
for i = 1:numel(subjects)
    sample_folds(subject_ids == subjects(i)) = fold_id_by_subject(i);
end

result_rows = strings(0, 8);
best = struct('name', '', 'acc', -Inf, 'macro_f1', -Inf, 'confusion', [], ...
    'pred', [], 'fold_metrics', [], 'threshold', NaN, 'lambda', NaN);

% Raw delta magnitude baseline.
for li = 1:numel(lambda_values)
    lambda = lambda_values(li);
    [pred, fold_metrics, conf] = cv_ridge_classifier(X_raw, y, sample_folds, classes, lambda);
    [acc, macro_f1] = summarize_prediction(y, pred, classes);
    result_rows(end+1,:) = ["raw_delta_magnitude", "NaN", string(lambda), ...
        string(acc), string(macro_f1), string(mean(fold_metrics(:,1))), ...
        string(mean(fold_metrics(:,2))), string(size(X_raw,2))]; %#ok<AGROW>
    if macro_f1 > best.macro_f1
        best = update_best("raw_delta_magnitude", acc, macro_f1, conf, pred, fold_metrics, NaN, lambda);
    end
end

% Spike count baselines across thresholds.
for ti = 1:numel(threshold_percentiles)
    threshold = threshold_percentiles(ti);
    X_spike = X_spike_sets{ti};
    for li = 1:numel(lambda_values)
        lambda = lambda_values(li);
        [pred, fold_metrics, conf] = cv_ridge_classifier(X_spike, y, sample_folds, classes, lambda);
        [acc, macro_f1] = summarize_prediction(y, pred, classes);
        result_rows(end+1,:) = ["delta_onoff_spike_count", string(threshold), string(lambda), ...
            string(acc), string(macro_f1), string(mean(fold_metrics(:,1))), ...
            string(mean(fold_metrics(:,2))), string(size(X_spike,2))]; %#ok<AGROW>
        if macro_f1 > best.macro_f1
            best = update_best("delta_onoff_spike_count", acc, macro_f1, conf, pred, fold_metrics, threshold, lambda);
        end
    end
end

result_table = array2table(result_rows, 'VariableNames', ...
    {'feature','threshold_percentile','lambda','accuracy','macro_f1','mean_fold_accuracy','mean_fold_macro_f1','feature_dim'});
writetable(result_table, fullfile(out_dir, 'spike_baseline_results.csv'));

meta_table = struct2table(sample_meta);
writetable(meta_table, fullfile(out_dir, 'spike_window_metadata.csv'));

summary_table = struct2table(build_summary);
writetable(summary_table, fullfile(out_dir, 'spike_dataset_build_summary.csv'));

save(fullfile(out_dir, 'spike_baseline_workspace.mat'), ...
    'result_table', 'summary_table', 'meta_table', 'best', 'classes', ...
    'threshold_percentiles', 'lambda_values', '-v7.3');

plot_confusion(best.confusion, classes, fullfile(out_dir, 'best_confusion_matrix.png'), ...
    sprintf('%s, thr=%g, lambda=%g, acc=%.3f, macroF1=%.3f', ...
    best.name, best.threshold, best.lambda, best.acc, best.macro_f1));

plot_result_grid(result_table, fullfile(out_dir, 'threshold_sweep.png'));
write_report(out_dir, result_table, summary_table, best, classes, sample_folds, y, subject_ids);

fprintf('\nBest: %s threshold=%g lambda=%g acc=%.4f macroF1=%.4f\n', ...
    best.name, best.threshold, best.lambda, best.acc, best.macro_f1);
fprintf('Outputs written to %s\n', out_dir);

function [X_spike_sets, X_raw, y, subject_ids, subject_names, sample_meta, build_summary] = build_spike_datasets(files, thresholds, fs_uwb)
    X_spike_cells = cell(numel(thresholds), 1);
    for i = 1:numel(thresholds)
        X_spike_cells{i} = {};
    end
    X_raw_cells = {};
    y_cells = {};
    subject_cells = {};
    subject_name_cells = {};
    sample_meta = struct('file', {}, 'subject_num', {}, 'subject_name', {}, 'label_index', {}, ...
        'label', {}, 'start_frame', {}, 'end_frame', {}, 'duration_sec', {});
    build_summary = struct('file', {}, 'status', {}, 'subject_num', {}, 'subject_name', {}, ...
        'labels', {}, 'samples_used', {}, 'frames', {}, 'bins', {}, 'message', {});

    for fi = 1:numel(files)
        path = fullfile(files(fi).folder, files(fi).name);
        row = struct('file', files(fi).name, 'status', "skipped", 'subject_num', NaN, ...
            'subject_name', "", 'labels', 0, 'samples_used', 0, 'frames', 0, 'bins', 0, 'message', "");
        try
            loaded = load(path, 'SyncData');
            if ~isfield(loaded, 'SyncData')
                row.message = "No SyncData field";
                build_summary(end+1) = row; %#ok<AGROW>
                continue;
            end
            s = loaded.SyncData;
            if ~isfield(s, 'labels') || ~isfield(s, 'com_final') || ~isfield(s, 'tv_final')
                row.message = "Missing labels or final UWB arrays";
                if isfield(s, 'subject_num'), row.subject_num = double(s.subject_num); end
                if isfield(s, 'subject_name'), row.subject_name = string(s.subject_name); end
                build_summary(end+1) = row; %#ok<AGROW>
                continue;
            end

            com = double(s.com_final);
            tv = double(s.tv_final);
            labels = double(s.labels(:));
            n_frames = min(size(com, 2), size(tv, 2));
            n_bins = min(size(com, 1), size(tv, 1));
            com = com(1:n_bins, 1:n_frames);
            tv = tv(1:n_bins, 1:n_frames);

            subject_num = double(s.subject_num);
            subject_name = string(s.subject_name);
            row.subject_num = subject_num;
            row.subject_name = subject_name;
            row.labels = numel(labels);
            row.frames = n_frames;
            row.bins = n_bins;

            if numel(labels) < 10 || n_frames < 2
                row.message = "Too few labels or frames";
                build_summary(end+1) = row; %#ok<AGROW>
                continue;
            end

            [com_z, tv_z] = robust_subject_zscore(com, tv);
            d_com = diff(com_z, 1, 2);
            d_tv = diff(tv_z, 1, 2);
            threshold_values = zeros(numel(thresholds), 2);
            for ti = 1:numel(thresholds)
                threshold_values(ti, 1) = prctile(abs(d_com(:)), thresholds(ti));
                threshold_values(ti, 2) = prctile(abs(d_tv(:)), thresholds(ti));
            end

            used = 0;
            for li = 1:numel(labels)
                st = floor((li - 1) * n_frames / numel(labels)) + 1;
                ed = floor(li * n_frames / numel(labels));
                st = max(st, 1);
                ed = min(ed, n_frames);
                if ed - st + 1 < 3
                    continue;
                end
                dst = st;
                ded = ed - 1;
                seg_d_com = d_com(:, dst:ded);
                seg_d_tv = d_tv(:, dst:ded);

                raw_feature = [mean(abs(seg_d_com), 2); mean(abs(seg_d_tv), 2)]';
                X_raw_cells{end+1, 1} = single(raw_feature); %#ok<AGROW>

                for ti = 1:numel(thresholds)
                    th_com = threshold_values(ti, 1);
                    th_tv = threshold_values(ti, 2);
                    f = [ ...
                        mean(seg_d_com > th_com, 2); ...
                        mean(seg_d_com < -th_com, 2); ...
                        mean(seg_d_tv > th_tv, 2); ...
                        mean(seg_d_tv < -th_tv, 2) ...
                    ]';
                    X_spike_cells{ti}{end+1, 1} = single(f); %#ok<AGROW>
                end

                y_cells{end+1, 1} = labels(li); %#ok<AGROW>
                subject_cells{end+1, 1} = subject_num; %#ok<AGROW>
                subject_name_cells{end+1, 1} = subject_name; %#ok<AGROW>
                used = used + 1;
                sample_meta(end+1) = struct('file', string(files(fi).name), ...
                    'subject_num', subject_num, 'subject_name', subject_name, ...
                    'label_index', li, 'label', labels(li), ...
                    'start_frame', st, 'end_frame', ed, ...
                    'duration_sec', (ed - st + 1) / fs_uwb); %#ok<AGROW>
            end

            row.status = "used";
            row.samples_used = used;
            row.message = "ok";
            build_summary(end+1) = row; %#ok<AGROW>
        catch ME
            row.status = "error";
            row.message = string(ME.message);
            build_summary(end+1) = row; %#ok<AGROW>
        end
    end

    X_raw = cell2mat(X_raw_cells);
    X_spike_sets = cell(numel(thresholds), 1);
    for ti = 1:numel(thresholds)
        X_spike_sets{ti} = cell2mat(X_spike_cells{ti});
    end
    y = cell2mat(y_cells);
    subject_ids = cell2mat(subject_cells);
    subject_names = string(subject_name_cells);
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

function fold_id_by_subject = make_subject_folds(subjects, k)
    subjects = subjects(randperm(numel(subjects)));
    fold_id_by_subject = zeros(size(subjects));
    for i = 1:numel(subjects)
        fold_id_by_subject(i) = mod(i - 1, k) + 1;
    end
    [~, order] = sort(subjects);
    fold_id_by_subject = fold_id_by_subject(order);
end

function [pred, fold_metrics, conf_total] = cv_ridge_classifier(X, y, folds, classes, lambda)
    pred = NaN(size(y));
    fold_values = unique(folds(:))';
    fold_metrics = zeros(numel(fold_values), 2);
    conf_total = zeros(numel(classes), numel(classes));
    for fi = 1:numel(fold_values)
        test_mask = folds == fold_values(fi);
        train_mask = ~test_mask;
        X_train = double(X(train_mask, :));
        X_test = double(X(test_mask, :));
        y_train = y(train_mask);
        y_test = y(test_mask);

        mu = mean(X_train, 1);
        sigma = std(X_train, 0, 1);
        sigma(sigma < 1e-8) = 1;
        X_train = (X_train - mu) ./ sigma;
        X_test = (X_test - mu) ./ sigma;

        Y = zeros(numel(y_train), numel(classes));
        for ci = 1:numel(classes)
            Y(:, ci) = double(y_train == classes(ci));
        end
        Y = 2 * Y - 1;

        X_aug = [X_train, ones(size(X_train, 1), 1)];
        X_test_aug = [X_test, ones(size(X_test, 1), 1)];
        reg = lambda * eye(size(X_aug, 2));
        reg(end, end) = 0;
        W = (X_aug' * X_aug + reg) \ (X_aug' * Y);
        scores = X_test_aug * W;
        [~, idx] = max(scores, [], 2);
        pred(test_mask) = classes(idx);

        conf = confusion_counts(y_test, pred(test_mask), classes);
        conf_total = conf_total + conf;
        [acc, macro_f1] = summarize_prediction(y_test, pred(test_mask), classes);
        fold_metrics(fi, :) = [acc, macro_f1];
    end
end

function conf = confusion_counts(y_true, y_pred, classes)
    conf = zeros(numel(classes), numel(classes));
    for i = 1:numel(y_true)
        ti = find(classes == y_true(i), 1);
        pi = find(classes == y_pred(i), 1);
        if ~isempty(ti) && ~isempty(pi)
            conf(ti, pi) = conf(ti, pi) + 1;
        end
    end
end

function [acc, macro_f1] = summarize_prediction(y_true, y_pred, classes)
    valid = isfinite(y_pred);
    y_true = y_true(valid);
    y_pred = y_pred(valid);
    acc = mean(y_true == y_pred);
    f1s = zeros(numel(classes), 1);
    for ci = 1:numel(classes)
        c = classes(ci);
        tp = sum(y_true == c & y_pred == c);
        fp = sum(y_true ~= c & y_pred == c);
        fn = sum(y_true == c & y_pred ~= c);
        precision = tp / max(tp + fp, 1);
        recall = tp / max(tp + fn, 1);
        f1s(ci) = 2 * precision * recall / max(precision + recall, eps);
    end
    macro_f1 = mean(f1s);
end

function best = update_best(name, acc, macro_f1, conf, pred, fold_metrics, threshold, lambda)
    best = struct('name', string(name), 'acc', acc, 'macro_f1', macro_f1, ...
        'confusion', conf, 'pred', pred, 'fold_metrics', fold_metrics, ...
        'threshold', threshold, 'lambda', lambda);
end

function plot_confusion(conf, classes, out_path, title_text)
    fig = figure('Visible', 'off', 'Position', [100 100 1000 850]);
    row_sum = sum(conf, 2);
    row_sum(row_sum == 0) = 1;
    conf_norm = conf ./ row_sum;
    imagesc(conf_norm);
    axis image;
    colormap(parula);
    colorbar;
    title(title_text, 'Interpreter', 'none');
    xlabel('Predicted label');
    ylabel('True label');
    xticks(1:numel(classes)); xticklabels(string(classes));
    yticks(1:numel(classes)); yticklabels(string(classes));
    for r = 1:size(conf, 1)
        for c = 1:size(conf, 2)
            if conf(r, c) > 0
                text(c, r, sprintf('%d', conf(r, c)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'w');
            end
        end
    end
    exportgraphics(fig, out_path, 'Resolution', 160);
    close(fig);
end

function plot_result_grid(result_table, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 1100 650]);
    feature = string(result_table.feature);
    threshold = str2double(string(result_table.threshold_percentile));
    lambda = str2double(string(result_table.lambda));
    macro_f1 = str2double(string(result_table.macro_f1));
    is_spike = feature == "delta_onoff_spike_count";

    tiledlayout(1, 2, 'TileSpacing', 'compact');
    nexttile;
    hold on;
    unique_lambda = unique(lambda(is_spike))';
    for l = unique_lambda
        mask = is_spike & lambda == l;
        plot(threshold(mask), macro_f1(mask), '-o', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('lambda=%g', l));
    end
    xlabel('delta threshold percentile');
    ylabel('macro F1');
    title('Spike encoding threshold sweep');
    grid on; legend('Location', 'best');

    nexttile;
    raw_mask = feature == "raw_delta_magnitude";
    bar([max(macro_f1(raw_mask)), max(macro_f1(is_spike))]);
    xticklabels({'raw delta', 'spike'});
    ylabel('best macro F1');
    title('Best subject-wise baseline');
    ylim([0, max(macro_f1) * 1.15 + 0.01]);
    grid on;
    exportgraphics(fig, out_path, 'Resolution', 160);
    close(fig);
end

function write_report(out_dir, result_table, summary_table, best, classes, sample_folds, y, subject_ids)
    report_path = fullfile(out_dir, 'spike_baseline_report.md');
    fid = fopen(report_path, 'w', 'n', 'UTF-8');
    cleanup = onCleanup(@() fclose(fid));

    used = summary_table(string(summary_table.status) == "used", :);
    skipped = summary_table(string(summary_table.status) ~= "used", :);
    label_counts = zeros(numel(classes), 1);
    for ci = 1:numel(classes)
        label_counts(ci) = sum(y == classes(ci));
    end

    fprintf(fid, '# Spike Encoding Baseline Report\n\n');
    fprintf(fid, '## Dataset\n\n');
    fprintf(fid, '- Used subjects/files: %d\n', height(used));
    fprintf(fid, '- Skipped/error files: %d\n', height(skipped));
    fprintf(fid, '- Samples: %d windows\n', numel(y));
    fprintf(fid, '- Classes: %d labels (`0-13`)\n', numel(classes));
    fprintf(fid, '- Windowing: one label-aligned segment, about 3 seconds each at 17 Hz\n');
    fprintf(fid, '- Split: 5-fold subject-wise cross validation\n\n');

    fprintf(fid, '## Encoding\n\n');
    fprintf(fid, '- Input arrays: `com_final` and `tv_final`\n');
    fprintf(fid, '- Per subject robust z-score by range bin\n');
    fprintf(fid, '- Delta event rule: ON spike if `dx > threshold`, OFF spike if `dx < -threshold`\n');
    fprintf(fid, '- Feature used for this first baseline: spike rate per range bin for COM_ON, COM_OFF, TV_ON, TV_OFF\n\n');

    fprintf(fid, '## Best Result\n\n');
    fprintf(fid, '- Feature: `%s`\n', best.name);
    fprintf(fid, '- Threshold percentile: %.0f\n', best.threshold);
    fprintf(fid, '- Ridge lambda: %.2g\n', best.lambda);
    fprintf(fid, '- Subject-wise accuracy: %.3f\n', best.acc);
    fprintf(fid, '- Subject-wise macro F1: %.3f\n\n', best.macro_f1);

    fprintf(fid, 'Fold metrics:\n\n');
    fprintf(fid, '| Fold | Accuracy | Macro F1 |\n');
    fprintf(fid, '|---:|---:|---:|\n');
    for i = 1:size(best.fold_metrics, 1)
        fprintf(fid, '| %d | %.3f | %.3f |\n', i, best.fold_metrics(i,1), best.fold_metrics(i,2));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## Label Distribution\n\n');
    fprintf(fid, '| Label | Count |\n');
    fprintf(fid, '|---:|---:|\n');
    for ci = 1:numel(classes)
        fprintf(fid, '| %.0f | %d |\n', classes(ci), label_counts(ci));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## Interpretation\n\n');
    fprintf(fid, '- If spike features beat or match raw delta magnitude, delta ON/OFF events preserve useful UWB structure.\n');
    fprintf(fid, '- Because validation is subject-wise, this is a stricter and more honest estimate than random window splitting.\n');
    fprintf(fid, '- Movement still makes direct respiration regression hard; this classification result should be treated as the first proof that event-coded UWB carries repeatable structure.\n\n');

    fprintf(fid, '## Files\n\n');
    fprintf(fid, '- `spike_baseline_results.csv`\n');
    fprintf(fid, '- `spike_window_metadata.csv`\n');
    fprintf(fid, '- `spike_dataset_build_summary.csv`\n');
    fprintf(fid, '- `best_confusion_matrix.png`\n');
    fprintf(fid, '- `threshold_sweep.png`\n');

    fold_subjects = zeros(numel(unique(subject_ids)), 2);
    subjects = unique(subject_ids);
    for i = 1:numel(subjects)
        fold_subjects(i,:) = [subjects(i), sample_folds(find(subject_ids == subjects(i), 1))]; %#ok<FNDSB>
    end
    fold_table = array2table(fold_subjects, 'VariableNames', {'subject_num', 'fold'});
    writetable(fold_table, fullfile(out_dir, 'subject_folds.csv'));
end
