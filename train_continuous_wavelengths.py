import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import gc
from sklearn.feature_selection import SelectFromModel


try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, total=None, desc=None, unit=None, leave=True):
        count = 0
        total = total or (len(iterable) if hasattr(iterable, '__len__') else None)
        for item in iterable or []:
            count += 1
            if total:
                print(f"{desc or ''} {count}/{total}", end='\r')
            else:
                print(f"{desc or ''} {count}", end='\r')
            yield item
        if leave:
            print()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_wavelengths(arg: str) -> List[int]:
    if isinstance(arg, str):
        if arg == 'all':
            return list(range(700, 901, 20))
        return [int(x) for x in arg.split(',') if x.strip()]
    return [int(x) for x in arg]


def list_patients(features_dir: str) -> List[str]:
    return [d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))]


def extract_pressure(name: str) -> int:
    m = re.search(r'pressure(\d+)', name)
    return int(m.group(1)) if m else None


def wl_remainder(wl: int) -> int:
    return ((wl - 780) // 20) % 11


def compute_roi_group(name: str, wl: int):
    bn = os.path.basename(str(name))
    if bn.endswith('.mat'):
        bn = bn[:-4]
    base = bn.split('__', 1)[0]
    m = re.match(r'(\d+)', base)
    if not m:
        return None
    try:
        base_id = int(m.group(1))
    except Exception:
        return None
    r = wl_remainder(wl)
    return (base_id - r) // 11


def build_rows_for_patient(features_dir: str, patient: str, wavelengths: List[int], normal_low: int, normal_high: int) -> List[Dict]:
    dfs = []
    roi_groups_list = []
    for wl in tqdm(wavelengths, desc=f'加载波长 {patient}', unit='wl', leave=False):
        fpath = os.path.join(features_dir, patient, f"{wl}nm.csv")
        if not os.path.isfile(fpath):
            return []
        df = pd.read_csv(fpath)
        if 'file' not in df.columns:
            return []
        base = df['file'].astype(str).str.replace('.mat', '', regex=False).str.split('__').str[0]
        base_id = base.str.extract(r'(\d+)')[0]
        base_id_num = pd.to_numeric(base_id, errors='coerce')
        r = wl_remainder(wl)
        roi_group = ((base_id_num - r) // 11).astype('Int64')
        df = df.assign(roi_group=roi_group)
        df = df[df['roi_group'].notna()].copy()
        df['roi_group'] = df['roi_group'].astype(np.int32)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != 'roi_group']
        if not num_cols:
            return []
        df_wl = df[['roi_group'] + num_cols].copy()
        for c in num_cols:
            try:
                df_wl[c] = pd.to_numeric(df_wl[c], errors='coerce').astype(np.float32)
            except Exception:
                pass
        df_wl.rename(columns={c: f"{wl}nm_{c}" for c in num_cols}, inplace=True)
        df_wl = df_wl.sort_values('roi_group').drop_duplicates(subset=['roi_group'], keep='first')
        dfs.append(df_wl)
        roi_groups_list.append(set(df_wl['roi_group'].tolist()))
    if not dfs:
        return []
    common_groups = set.intersection(*roi_groups_list) if roi_groups_list else set()
    if not common_groups:
        return []
    for i in tqdm(range(len(dfs)), desc='过滤公共ROI', unit='df', leave=False):
        dfs[i] = dfs[i][dfs[i]['roi_group'].isin(common_groups)]
    df_join = dfs[0]
    for k in range(1, len(dfs)):
        df_join = pd.merge(df_join, dfs[k], on='roi_group', how='inner', sort=False)
        dfs[k] = None
    if df_join.empty:
        return []
    pressure = extract_pressure(patient)
    is_normal_gt = int(normal_low <= pressure <= normal_high)
    df_out = df_join.copy()
    df_out['patient'] = patient
    df_out['pressure'] = pressure
    df_out['is_normal_gt'] = is_normal_gt
    df_out['file'] = df_out['roi_group'].astype(int).astype(str)
    rows = df_out.to_dict(orient='records')
    del dfs, df_join
    gc.collect()
    return rows


def consecutive_windows(wavelengths: List[int], window_size: int, step: int = 1) -> List[List[int]]:
    wls = sorted(wavelengths)
    wins = []
    for i in range(0, max(0, len(wls) - window_size + 1), step):
        seg = wls[i:i + window_size]
        if len(seg) < window_size:
            continue
        if all((seg[k + 1] - seg[k]) == 20 for k in range(len(seg) - 1)):
            wins.append(seg)
    return wins


def eligible_patients_for_window(features_dir: str, wavelengths: List[int]) -> List[str]:
    pts = []
    for p in list_patients(features_dir):
        ok = True
        for wl in wavelengths:
            if not os.path.isfile(os.path.join(features_dir, p, f"{wl}nm.csv")):
                ok = False
                break
        if ok:
            pts.append(p)
    return pts


def stratified_split_by_patient(patients: List[str], normal_low: int, normal_high: int, test_size: float, random_state: int, label_mode: str = 'normal_abnormal') -> Tuple[List[str], List[str]]:
    if label_mode == 'high_low':
        labels = [1 if extract_pressure(p) > normal_high else 0 for p in patients]
    else:
        labels = [int(normal_low <= extract_pressure(p) <= normal_high) for p in patients]
    train_pts, test_pts = train_test_split(patients, test_size=test_size, random_state=random_state, stratify=labels)
    return train_pts, test_pts


def main():
    ap = argparse.ArgumentParser(description='Train SVM on ROI-level features with consecutive wavelength windows')
    ap.add_argument('--features-dir', default=str(Path(__file__).resolve().parent / 'fuse_qmad_sato'))
    ap.add_argument('--models-dir', default=str(Path(__file__).resolve().parent / 'models_cont'))
    ap.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'results_cont'))
    ap.add_argument('--wavelengths', default='all')
    ap.add_argument('--window-size', type=int, default=3)
    ap.add_argument('--window-step', type=int, default=1)
    ap.add_argument('--normal-range', default='140,190')
    ap.add_argument('--label-mode', default='high_low', choices=['high_low', 'normal_abnormal'])
    ap.add_argument('--test-size', type=float, default=0.3)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--patients', default='')
    ap.add_argument('--patients-file', default='')
    ap.add_argument('--use-pca', action='store_true', default=False)
    ap.add_argument('--pca-components', type=float, default=0.95)
    ap.add_argument('--scaler', default='standard', choices=['standard','robust'])
    ap.add_argument('--feature-select', default='none', choices=['none','l1'])
    ap.add_argument('--l1-C', type=float, default=0.5)
    ap.add_argument('--l1-max-iter', type=int, default=5000)
    ap.add_argument('--l1-tol', type=float, default=1e-4)
    ap.add_argument('--svc-C', type=float, default=1.0)
    ap.add_argument('--svc-gamma', default='scale')
    ap.add_argument('--svc-class-weight', default='none')  
    ap.add_argument('--svc-cache-size', type=float, default=200)
    args = ap.parse_args()
    label_mode = args.label_mode
    
    features_dir = args.features_dir
    models_dir = args.models_dir
    output_dir = args.output_dir
    wavelengths = parse_wavelengths(args.wavelengths)
    if isinstance(args.normal_range, str):
        try:
            normal_low, normal_high = [int(x) for x in args.normal_range.split(',')]
        except Exception:
            normal_low, normal_high = 140, 190
    else:
        normal_low, normal_high = 140, 190
    test_size = args.test_size
    random_state = args.random_state

    ensure_dir(models_dir)
    ensure_dir(output_dir)

    windows = consecutive_windows(wavelengths, args.window_size, args.window_step)
    if not windows:
        print('[ERROR] 无有效的连续波长窗口。请调整 --window-size 或提供正确的 --wavelengths')
        return

    json_train: List[str] = []
    json_test: List[str] = []
    all_patients: List[str] = []
    if args.patients_file and os.path.isfile(args.patients_file):
        try:
            with open(args.patients_file, 'r') as f:
                sp = json.load(f)
            if isinstance(sp, dict):
                if 'train' in sp and isinstance(sp['train'], list):
                    json_train = sp['train']
                if 'test' in sp and isinstance(sp['test'], list):
                    json_test = sp['test']
                if 'patients' in sp and isinstance(sp['patients'], list):
                    all_patients = sp['patients']
        except Exception:
            json_train, json_test, all_patients = [], [], []
    if not all_patients and args.patients:
        all_patients = [s for s in args.patients.split(',') if s.strip()]
    if not all_patients:
        all_patients = list_patients(features_dir)

    overall_summary = []
    for win in tqdm(windows, desc='训练窗口', unit='窗口'):
        patients_win = [p for p in all_patients if all(os.path.isfile(os.path.join(features_dir, p, f"{wl}nm.csv")) for wl in win)]
        wl_tag = f"{win[0]}-{win[-1]}"
        if json_train or json_test:
            train_patients = [p for p in patients_win if (not json_train or p in json_train)]
            test_patients = [p for p in patients_win if (json_test and p in json_test)]
            if label_mode == 'high_low':
                train_patients = [p for p in train_patients if (extract_pressure(p) < normal_low) or (extract_pressure(p) > normal_high)]
                test_patients = [p for p in test_patients if (extract_pressure(p) < normal_low) or (extract_pressure(p) > normal_high)]
            if not train_patients or not test_patients:
                print(f"[SKIP] 窗口 {wl_tag} JSON 划分下可用患者不足，跳过。")
                continue
            print(f"[TRAIN] 连续窗口 {wl_tag}，使用 JSON 划分，训练数={len(train_patients)}，测试数={len(test_patients)}，波长数量={len(win)}")
        else:
            if label_mode == 'high_low':
                patients_win = [p for p in patients_win if (extract_pressure(p) < normal_low) or (extract_pressure(p) > normal_high)]
            if len(patients_win) < 2:
                print(f"[SKIP] 窗口 {wl_tag} 可用患者不足，跳过。")
                continue
            print(f"[TRAIN] 连续窗口 {wl_tag}，波长数量={len(win)}，患者数={len(patients_win)}")
            try:
                train_patients, test_patients = stratified_split_by_patient(patients_win, normal_low, normal_high, test_size, random_state, label_mode)
            except Exception:
                split_idx = max(1, int(len(patients_win) * (1 - test_size)))
                train_patients = patients_win[:split_idx]
                test_patients = patients_win[split_idx:]

        rows_train = []
        for p in tqdm(train_patients, desc=f'构建训练样本 {wl_tag}', leave=False):
            rs = build_rows_for_patient(features_dir, p, win, normal_low, normal_high)
            rows_train.extend(rs)
        rows_test = []
        for p in tqdm(test_patients, desc=f'构建测试样本 {wl_tag}', leave=False):
            rs = build_rows_for_patient(features_dir, p, win, normal_low, normal_high)
            rows_test.extend(rs)

        if not rows_train or not rows_test:
            print(f"[SKIP] 窗口 {wl_tag} 无足够训练或测试样本，跳过。")
            continue

        df_train = pd.DataFrame(rows_train)
        df_test = pd.DataFrame(rows_test)
        if label_mode == 'high_low':
            df_train = df_train[(df_train['pressure'] < normal_low) | (df_train['pressure'] > normal_high)].copy()
            df_test = df_test[(df_test['pressure'] < normal_low) | (df_test['pressure'] > normal_high)].copy()
            df_train['is_high_gt'] = (df_train['pressure'] > normal_high).astype(int)
            df_test['is_high_gt'] = (df_test['pressure'] > normal_high).astype(int)
        if df_train.empty or df_test.empty:
            print(f"[SKIP] 窗口 {wl_tag} 过滤后无样本，跳过。")
            continue

        num_cols_train = df_train.select_dtypes(include=[np.number]).columns.tolist()
        label_col = 'is_high_gt' if label_mode == 'high_low' else 'is_normal_gt'
        feature_cols = [c for c in num_cols_train if c not in [label_col, 'pressure', 'roi_group']]
        for c in feature_cols:
            if c not in df_test.columns:
                print(f"[SKIP] 测试集中缺少特征列 {c}，窗口 {wl_tag}，跳过。")
                feature_cols = []
                break
        if not feature_cols:
            continue

        X_train = df_train[feature_cols].values
        y_train = df_train[label_col].astype(int).values
        X_test = df_test[feature_cols].values
        y_test = df_test[label_col].astype(int).values

        def parse_class_weight_arg(s: str):
            if not s or s == 'none':
                return None
            if s == 'balanced':
                return 'balanced'
            try:
                parts = [p.strip() for p in s.split(',') if p.strip()]
                d = {}
                for p in parts:
                    k, v = p.split(':')
                    d[int(k.strip())] = float(v.strip())
                return d
            except Exception:
                return None

        steps = []
        # 1）先进行sacle
        scaler = RobustScaler() if args.scaler == 'robust' else StandardScaler()
        steps.append(('scaler', scaler))

        # 2）特征选择（L1）
        selected_feature_names = None
        if args.feature_select == 'l1':
            l1 = LinearSVC(penalty='l1', dual=False, C=args.l1_C, class_weight='balanced', max_iter=args.l1_max_iter, tol=args.l1_tol, random_state=random_state)

             # 使用 SelectFromModel；指定 threshold（例如 "median" 或 explicit）
            steps.append(('sfm', SelectFromModel(estimator=l1, threshold='median')))
        
        # 3）PCA降维
        if args.use_pca:
            steps.append(('pca', PCA(n_components=args.pca_components)))

        # 4）分类器或 Dummy
        if len(np.unique(y_train)) < 2:
            const_class = int(np.unique(y_train)[0]) if len(y_train) > 0 else 0
            steps.append(('dummy', DummyClassifier(strategy='constant', constant=const_class)))
        else:
            cw = parse_class_weight_arg(args.svc_class_weight)
            gamma = args.svc_gamma
            try:
                gamma = float(gamma)
            except Exception:
                pass
            steps.append(('svc', SVC(C=args.svc_C, kernel='rbf', gamma=gamma, class_weight=cw, probability=True, cache_size=args.svc_cache_size, random_state=random_state)))
        
        pipe = Pipeline(steps)
        pipe.fit(X_train, y_train)

        # ---------- 获取被选中特征的名称（如果使用了 SelectFromModel） ----------
        selected_feature_names = None
        if args.feature_select == 'l1':
            # pipeline.named_steps['sfm'] 在 scaler 之后，因此需要取被选中列的 mask
            sfm = pipe.named_steps.get('sfm', None)
            if sfm is not None:
                # support_ 是布尔掩码
                mask = sfm.get_support()
                selected_feature_names = [f for f, m in zip(feature_cols, mask) if m]
        else:
            selected_feature_names = feature_cols[:]  # 全部特征未筛选时保留完整列表

        y_pred_tr = pipe.predict(X_train)
        acc_tr = float(accuracy_score(y_train, y_pred_tr))
        f1_tr = float(f1_score(y_train, y_pred_tr, average='macro'))
        cm_tr = confusion_matrix(y_train, y_pred_tr).tolist()

        y_pred_te = pipe.predict(X_test)
        acc_te = float(accuracy_score(y_test, y_pred_te))
        f1_te = float(f1_score(y_test, y_pred_te, average='macro'))
        cm_te = confusion_matrix(y_test, y_pred_te).tolist()
        report_te = classification_report(y_test, y_pred_te, output_dict=True, zero_division=0)

        model_obj = {
            'pipeline': pipe,
            'feature_cols': feature_cols,
            'selected_features': selected_feature_names,
            'wavelengths': win,
            'normal_range': [normal_low, normal_high],
            'window_tag': wl_tag,
            'pca': {
                'enabled': bool(args.use_pca),
                'n_components': args.pca_components if args.use_pca else None
            }
        }
        model_path = os.path.join(models_dir, f'svm_cont_{wl_tag}.joblib')
        joblib.dump(model_obj, model_path)

        metrics = {
            'window': win,
            'n_train_rows': len(df_train),
            'n_test_rows': len(df_test),
            'normal_range': [normal_low, normal_high],
            'pca_enabled': bool(args.use_pca),
            'pca_components': args.pca_components if args.use_pca else None,
            'train': {
                'accuracy': acc_tr,
                'f1_macro': f1_tr,
                'confusion_matrix': cm_tr
            },
            'test': {
                'accuracy': acc_te,
                'f1_macro': f1_te,
                'confusion_matrix': cm_te,
                'classification_report': report_te
            }
        }
        metrics_path = os.path.join(output_dir, f'metrics_cont_svm_{wl_tag}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        def predict_rows(df: pd.DataFrame) -> List[Dict]:
            X = df[feature_cols].values
            y_gt_each = df[label_col].astype(int).values
            pos_name = 'high' if label_mode == 'high_low' else 'normal'
            neg_name = 'low' if label_mode == 'high_low' else 'abnormal'
            pos_prob_name = 'prob_high' if label_mode == 'high_low' else 'prob_normal'
            neg_prob_name = 'prob_low' if label_mode == 'high_low' else 'prob_abnormal'
            try:
                proba = pipe.predict_proba(X)
                classes = list(pipe.classes_) if hasattr(pipe, 'classes') else [0, 1]
                idx_pos = classes.index(1) if 1 in classes else 1
                idx_neg = classes.index(0) if 0 in classes else 0
                p_pos_each = [float(proba[i, idx_pos]) for i in range(proba.shape[0])]
                p_neg_each = [float(proba[i, idx_neg]) for i in range(proba.shape[0])]
            except Exception:
                p_pos_each = [None] * len(y_gt_each)
                p_neg_each = [None] * len(y_gt_each)
            y_pred_each = pipe.predict(X)
            rows_out = []
            for i in range(len(y_pred_each)):
                pred_label = pos_name if int(y_pred_each[i]) == 1 else neg_name
                gt_label = pos_name if int(y_gt_each[i]) == 1 else neg_name
                rows_out.append({
                    'patient': str(df.iloc[i]['patient']),
                    'file': str(df.iloc[i]['file']) if 'file' in df.columns else '',
                    'pressure': int(df.iloc[i]['pressure']),
                    'gt': gt_label,
                    'pred': pred_label,
                    pos_prob_name: p_pos_each[i],
                    neg_prob_name: p_neg_each[i],
                    'correct': int(int(y_pred_each[i]) == int(y_gt_each[i]))
                })
            return rows_out

        rows_tr_pred = predict_rows(df_train)
        rows_te_pred = predict_rows(df_test)
        pd.DataFrame(rows_tr_pred).to_csv(os.path.join(output_dir, f'predictions_cont_svm_{wl_tag}_train_roi.csv'), index=False)
        pd.DataFrame(rows_te_pred).to_csv(os.path.join(output_dir, f'predictions_cont_svm_{wl_tag}_test_roi.csv'), index=False)

        def summarize_patient(rows: List[Dict]) -> List[Dict]:
            grouped: Dict[str, List[Dict]] = {}
            for r in rows:
                grouped.setdefault(r['patient'], []).append(r)
            out = []
            pos_prob_name = 'prob_high' if label_mode == 'high_low' else 'prob_normal'
            neg_prob_name = 'prob_low' if label_mode == 'high_low' else 'prob_abnormal'
            for p, rs in grouped.items():
                preds = [r['pred'] for r in rs]
                overall_pred = max(set(preds), key=preds.count)
                pn = [r.get(pos_prob_name) for r in rs if r.get(pos_prob_name) is not None]
                pa = [r.get(neg_prob_name) for r in rs if r.get(neg_prob_name) is not None]
                avg_pn = float(np.mean(pn)) if pn else None
                avg_pa = float(np.mean(pa)) if pa else None
                gt_label = rs[0]['gt']
                correct = int((overall_pred == gt_label))
                out.append({
                    'patient': p,
                    'pressure': rs[0]['pressure'],
                    'gt': gt_label,
                    'pred': overall_pred,
                    pos_prob_name: avg_pn,
                    neg_prob_name: avg_pa,
                    'correct': correct
                })
            return out

        sum_tr = summarize_patient(rows_tr_pred)
        sum_te = summarize_patient(rows_te_pred)
        pd.DataFrame(sum_tr).to_csv(os.path.join(output_dir, f'predictions_cont_svm_{wl_tag}_train.csv'), index=False)
        pd.DataFrame(sum_te).to_csv(os.path.join(output_dir, f'predictions_cont_svm_{wl_tag}_test.csv'), index=False)

        pos_name = 'high' if label_mode == 'high_low' else 'normal'
        y_true = [1 if r['gt'] == pos_name else 0 for r in sum_te]
        y_pred = [1 if r['pred'] == pos_name else 0 for r in sum_te]
        acc = float(accuracy_score(y_true, y_pred)) if y_true else None
        f1m = float(f1_score(y_true, y_pred, average='macro')) if y_true else None
        cm = confusion_matrix(y_true, y_pred).tolist() if y_true else []
        overall_summary.append({
            'window_tag': wl_tag,
            'wavelengths': win,
            'test_patients': len(sum_te),
            'accuracy': acc,
            'f1_macro': f1m,
            'confusion_matrix': cm
        })
        del df_train, df_test, rows_train, rows_test, rows_tr_pred, rows_te_pred, sum_tr, sum_te
        gc.collect()

    if overall_summary:
        with open(os.path.join(output_dir, 'overall_windows_summary.json'), 'w') as f:
            json.dump(overall_summary, f, ensure_ascii=False, indent=2)
        print(f"[DONE] 已保存 {len(overall_summary)} 个窗口的训练与评估结果。")
    else:
        print('[WARN] 没有可用的窗口训练结果。')


    print(f"[TRAIN] {wl_tag}: 训练集 Acc={acc_tr:.3f}, F1={f1_tr:.3f}")
    print(f"[TEST] {wl_tag}: 测试集 Acc={acc_te:.3f}, F1={f1_te:.3f}")

if __name__ == '__main__':
    main()