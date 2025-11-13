#!/usr/bin/env python3
import os, json, argparse, subprocess, shutil, time
from pathlib import Path
import csv

HERE = Path(__file__).resolve().parent
TRAIN_PY = HERE / 'train_continuous_wavelengths.py'

DEF_MODELS = HERE / 'models_cont'
DEF_RESULTS = HERE / 'results_cont'
DEF_ARCHIVES = HERE / 'archives'

WL_ALL = [700,720,740,760,780,800,820,840,860,880,900]

CLASS_WEIGHT_MAP = {
    'balanced': 'balanced',
    'none': 'none'
}

def run_cmd(cmd):
    print('RUN:', ' '.join(cmd))
    r = subprocess.run(cmd, check=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description='Read JSON config and reproduce training snapshot')
    ap.add_argument('--config', required=True)
    ap.add_argument('--models-dir', default=str(DEF_MODELS))
    ap.add_argument('--results-dir', default=str(DEF_RESULTS))
    ap.add_argument('--archives-dir', default=str(DEF_ARCHIVES))
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    features_dir = cfg.get('features_dir') or str(HERE / 'fuse_qmad_sato')
    wls = cfg.get('wavelengths', 'all')
    if isinstance(wls, str) and wls == 'all':
        wls = WL_ALL
    wl_arg = ','.join(str(x) for x in wls)
    ws = int(cfg.get('window_size', 11))
    step = int(cfg.get('window_step', 11))
    label_mode = cfg.get('label_mode', 'normal_abnormal')
    normal_range = cfg.get('normal_range', [140,190])
    rs = int(cfg.get('random_state', 42))


    scaler = cfg.get('scaler', 'standard')
    feature_select = cfg.get('feature_select', 'none')
    l1_C = float(cfg.get('l1_C', 0.5))
    svc_kernel = cfg.get('svc_kernel', 'rbf')
    svc_probability = bool(cfg.get('svc_probability', True))
    svc_C = float(cfg.get('svc_C', 2.0)) if 'svc_C' in cfg else 2.0
    svc_gamma = cfg.get('svc_gamma', 0.003) if 'svc_gamma' in cfg else 0.003
    svc_class_weight = cfg.get('svc_class_weight', '0:1,1:2') if 'svc_class_weight' in cfg else '0:1,1:2'
    svc_cache_size = float(cfg.get('svc_cache_size', 512))

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.archives_dir).mkdir(parents=True, exist_ok=True)


    wl_tag = f"{wls[0]}-{wls[-1]}"
    cfg_dir = Path(args.config).parent
    pred_train = cfg_dir / f'predictions_cont_svm_{wl_tag}_train.csv'
    pred_test = cfg_dir / f'predictions_cont_svm_{wl_tag}_test.csv'
    patients_file = None
    if pred_train.exists() and pred_test.exists():
        pts_train = set()
        pts_test = set()
        with open(pred_train, 'r') as f:
            r = csv.DictReader(f)
            for row in r:
                if 'patient' in row and row['patient']:
                    pts_train.add(row['patient'])
        with open(pred_test, 'r') as f:
            r = csv.DictReader(f)
            for row in r:
                if 'patient' in row and row['patient']:
                    pts_test.add(row['patient'])
        if pts_train and pts_test:
            patients_file = Path(args.results_dir) / f'patients_split_{wl_tag}.json'
            with open(patients_file, 'w') as f:
                json.dump({
                    'train': sorted(list(pts_train)),
                    'test': sorted(list(pts_test)),
                    'patients': sorted(list(pts_train.union(pts_test)))
                }, f, ensure_ascii=False, indent=2)

    cmd = [
        'python3', str(TRAIN_PY),
        '--features-dir', str(features_dir),
        '--models-dir', str(args.models_dir),
        '--output-dir', str(args.results_dir),
        '--wavelengths', wl_arg,
        '--window-size', str(ws),
        '--window-step', str(step),
        '--label-mode', label_mode,
        '--normal-range', f"{normal_range[0]},{normal_range[1]}",
        '--random-state', str(rs),
        '--test-size','0.3',
        '--scaler', scaler if scaler in ['standard','robust'] else 'standard',
        '--feature-select', feature_select if feature_select in ['none','l1'] else 'none',
        '--l1-C', str(l1_C),
        '--svc-C', str(svc_C),
        '--svc-gamma', str(svc_gamma),
        '--svc-class-weight', svc_class_weight,
        '--svc-cache-size', str(svc_cache_size)
    ]
    if cfg.get('use_pca', False):
        cmd += ['--use-pca','--pca-components', str(cfg.get('pca_components', 0.95))]
    if patients_file:
        cmd += ['--patients-file', str(patients_file)]
    run_cmd(cmd)

    # Locate window outputs
    files = [
        f'svm_cont_{wl_tag}.joblib',
        f'metrics_cont_svm_{wl_tag}.json',
        f'predictions_cont_svm_{wl_tag}_test.csv',
        f'predictions_cont_svm_{wl_tag}_test_roi.csv',
        f'predictions_cont_svm_{wl_tag}_train.csv',
        f'predictions_cont_svm_{wl_tag}_train_roi.csv',
    ]

    ts = time.strftime('%Y%m%d_%H%M%S')
    snap_dir = Path(args.archives_dir) / f'svm_from_config_{wl_tag}_{ts}'
    snap_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for fn in files:
        src = Path(args.results_dir) / fn if fn.startswith('metrics') or fn.startswith('predictions') else Path(args.models_dir) / fn
        if src.exists():
            dst = snap_dir / fn
            shutil.copy2(src, dst)
            copied.append(str(dst))
        else:
            print('[WARN] missing:', src)


    summary = {}
    mpath = Path(args.results_dir) / f'metrics_cont_svm_{wl_tag}.json'
    if mpath.exists():
        try:
            m = json.load(open(mpath))
            summary = {
                'accuracy': m.get('test',{}).get('accuracy'),
                'f1_macro': m.get('test',{}).get('f1_macro'),
                'confusion_matrix': m.get('test',{}).get('confusion_matrix')
            }
        except Exception:
            pass
    out_cfg = dict(cfg)
    out_cfg['timestamp'] = ts
    out_cfg['files'] = [os.path.basename(p) for p in copied]
    out_cfg['metrics_summary'] = summary
    with open(snap_dir / 'train_config.json','w') as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)
    print('[DONE] snapshot at', str(snap_dir))

if __name__ == '__main__':
    main()
