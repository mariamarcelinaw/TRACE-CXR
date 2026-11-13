import csv
import json
import tempfile
import unittest
from pathlib import Path
from pipeline import auc, binary_metrics, sigmoid, logit, calibrate, gate_features, fuse, reliability_bins, ece_from_bins
from audit import require_disjoint_by_role, check_split_csv, verify_manifest, verify_frozen_roles
from reproduce import main


class ResearchReproductionTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'configs/historical.json').read_text())

    def test_auc_perfect(self):self.assertAlmostEqual(auc([0,1,0,1],[.1,.9,.2,.8]),1)
    def test_auc_with_tie(self):self.assertAlmostEqual(auc([0,1],[.5,.5]),.5)
    def test_auc_invalid_single_class(self):
        with self.assertRaises(ValueError):auc([1,1],[.1,.2])
    def test_metrics_uses_frozen_threshold(self):
        r=binary_metrics([1,0,1,0],[.9,.1,.2,.8],.5)
        self.assertEqual((r['tp'],r['tn'],r['fp'],r['fn']),(1,1,1,1))
    def test_reliability_bins_and_ece(self):
        b=reliability_bins([0,1,1,0],[0.05,0.05,0.95,0.95])
        self.assertEqual(sum(x['n_images'] for x in b),4)
        self.assertAlmostEqual(ece_from_bins(b,4),0.45)
    def test_reliability_evidence_matches_reported_ece(self):
        root=Path(__file__).resolve().parents[1]
        with (root/'evidence/reliability_bins.csv').open() as f:bins=list(csv.DictReader(f))
        with (root/'evidence/calibration_summary.csv').open() as f:
            summary={(r['cohort'],r['model']):r for r in csv.DictReader(f)}
        for (cohort,model),row in summary.items():
            sel=[r for r in bins if r['cohort']==cohort and r['model']==model and int(r['n_images'])]
            n=int(row['n_images']);self.assertEqual(sum(int(r['n_images']) for r in sel),n)
            ece=sum(int(r['n_images'])/n*abs(float(r['observed_rate'])-float(r['mean_predicted'])) for r in sel)
            self.assertAlmostEqual(ece,float(row['ece15']),places=3)
        self.assertAlmostEqual(float(summary[('KermanyNorm','gated_fusion')]['ece15']),0.3898,places=4)
        self.assertAlmostEqual(float(summary[('RSNA locked test','gated_fusion')]['ece15']),0.0209,places=4)
    def test_temperature_identity(self):self.assertAlmostEqual(calibrate(.2,1),.2)
    def test_logit_sigmoid(self):self.assertAlmostEqual(sigmoid(logit(.92)),.92)
    def test_gate_feature_order(self):
        f=gate_features(.2,.8); self.assertEqual(len(f),5); self.assertAlmostEqual(f[-1],.6)
    def test_mean_fusion(self):self.assertAlmostEqual(fuse(.4,.9,'mean_fusion',self.config['control_lock']),.65)
    def test_gated_without_alpha_rejected(self):
        with self.assertRaises(ValueError):fuse(.4,.9,'gated_fusion',self.config['control_lock'])
    def test_const_alpha_uses_control(self):
        p=.4;s=.9;a=self.config['control_lock']['const_alpha_inception']
        self.assertAlmostEqual(fuse(p,s,'const_alpha',self.config['control_lock']),a*p+(1-a)*s)
    def test_disjoint(self):require_disjoint_by_role({'dev':{'a'},'test':{'b'}})
    def test_leakage_fails(self):
        with self.assertRaises(ValueError):require_disjoint_by_role({'dev':{'x'},'locked_test':{'x'}})
    def test_false_external_role_fails(self):
        config=json.loads(json.dumps(self.config)); config['fitting_roles']['external_kermany']='calibration'
        with self.assertRaises(ValueError):verify_frozen_roles(config)
    def test_split_duplicate_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'split.csv';p.write_text('id,split,label,fold\nx,development,1,0\nx,locked_test,1,\n')
            with self.assertRaises(ValueError):check_split_csv(p)
    def test_split_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'split.csv';p.write_text('id,split,label,fold\na,development,1,0\nb,calibration,0,\nc,locked_test,1,\n')
            self.assertEqual(check_split_csv(p)['n_total'],3)
    def test_public_split_manifest_matches_paper(self):
        root=Path(__file__).resolve().parents[1]
        r=check_split_csv(root/'data/rsna_split_manifest.csv')
        self.assertEqual(r['split_counts'],{'calibration':2669,'development':18678,'locked_test':5337})
        self.assertEqual(sorted(r['development_folds'].values()),[3735,3735,3736,3736,3736])
    def test_split_without_label_column_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'split.csv';p.write_text('id,split,fold\na,development,0\nb,calibration,\nc,locked_test,\n')
            self.assertEqual(check_split_csv(p)['n_total'],3)
    def test_recalibration_summary_matches_paper_n20(self):
        root=Path(__file__).resolve().parents[1]
        with open(root/'evidence/recalibration_draw_summary.csv',encoding='utf-8') as fh: rows=list(csv.DictReader(fh))
        r=[x for x in rows if x['model']=='gated_fusion' and x['strategy']=='stratified' and x['method']=='full_logistic' and x['n']=='20'][0]
        self.assertEqual((round(float(r['ECE15_p2_5']),3),round(float(r['ECE15_p97_5']),3)),(0.054,0.159))
        self.assertEqual(r['draws_success'],'500')
    def test_primary_run_distribution_matches_table_iv(self):
        root=Path(__file__).resolve().parents[1]
        with open(root/'evidence/recalibration_primary_run_distribution.csv',encoding='utf-8') as fh: rows=list(csv.DictReader(fh))
        r={x['n']:x for x in rows if x['model']=='gated_fusion'}
        self.assertEqual([round(float(r[n]['ECE15_mean']),3) for n in ('10','20','50','100','200')],[0.120,0.093,0.072,0.063,0.062])
        self.assertEqual((r['10']['nonpositive_slopes'],r['20']['nonpositive_slopes'],r['50']['nonpositive_slopes']),('1','1','0'))
    def test_patient_overlap_result_matches_paper(self):
        root=Path(__file__).resolve().parents[1]
        text=(root/'evidence/patient_overlap_result.txt').read_text()
        for snippet in ('NIH patients in all 26684 images: 11452','patient also in development 3645 (68.3%)',
                        'n=1598 images, 1497 patients','dAUC gate - swin_b        +0.0103 [+0.0022, +0.0197]'):
            self.assertIn(snippet,text)
    def test_public_checksum(self):
        result=verify_manifest(Path(__file__).resolve().parents[1]);self.assertGreater(result['files_verified'],0)
    def test_cli_help(self):
        # argparse help exit=0, calls no training
        with self.assertRaises(SystemExit) as exc:main(['--help'])
        self.assertEqual(exc.exception.code,0)
    def test_cli_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(['verify','--output-root',tmp]),0)
    def test_cli_all_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(['all','--output-root',tmp]),0)
            content=json.loads((Path(tmp)/'trace_cxr_all.json').read_text())
            self.assertFalse(content['gpu_used']);self.assertFalse(content['training_performed'])

    def test_license_is_mit(self):
        text=(Path(__file__).resolve().parents[1]/'LICENSE').read_text()
        self.assertTrue(text.startswith('MIT License'))
        self.assertEqual(self.config['authorization']['repository_license'],'MIT')

    def test_both_kermany_preprocessing_versions_reported(self):
        v=self.config['archived_kermany_preprocessing_versions']
        self.assertEqual(v['primary'],'KermanyNorm')
        for version in ('KermanyRaw','KermanyNorm'):
            self.assertEqual(set(v['roc_auc'][version]),set(self.config['candidate_names']))
        self.assertLess(v['roc_auc']['KermanyNorm']['gated_fusion'],v['roc_auc']['KermanyRaw']['gated_fusion'])

    def test_adaptation_threshold_rules_declared(self):
        rules=self.config['retrospective_adaptation_rules']['threshold_rule']
        self.assertEqual(rules['intercept_only'],'frozen RSNA threshold')
        self.assertEqual(rules['full_logistic'],'labeled-subset prevalence')

    def _write_timing(self, path, means=None):
        means=means or self.config['archived_latency_benchmark']['candidate_means_ms']
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['candidate','total_end_to_end_ms'])
            for c in self.config['candidate_names']:
                for i in range(600):w.writerow([c,float(means[c])+(1 if i%2 else -1)])

    def _run_timing(self, tmp):
        out=Path(tmp)/'out'
        code=main(['timing','--data-root',tmp,'--output-root',str(out)])
        content=json.loads((out/'trace_cxr_timing.json').read_text()) if code==0 else None
        return code,content

    def test_interleaved_timing_file_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'INTERLEAVED_LATENCY_RAW_TIMINGS.csv';self._write_timing(path)
            code,content=self._run_timing(tmp)
            self.assertEqual(code,0)
            self.assertEqual(content['results']['timing']['source'],path.name)
            self.assertEqual(content['results']['timing']['status'],'PASS_INTERLEAVED_TIMING_RECOMPUTED_FROM_LOCAL_ROWS')

    def test_prefixed_archived_timing_file_preferred_over_older_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            archived=Path(tmp)/'ARCHIVE_RUN_SIX_CANDIDATE_RAW_TIMINGS.csv';self._write_timing(archived)
            (Path(tmp)/'TRACE_CXR_INFERENCE_TIME.csv').write_text('total\n134.8\n135.0\n')
            code,content=self._run_timing(tmp)
            self.assertEqual(code,0)
            self.assertEqual(content['results']['timing']['source'],archived.name)

    def test_older_timing_record_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'TRACE_CXR_INFERENCE_TIME.csv').write_text('total\n134.8\n135.0\n')
            code,content=self._run_timing(tmp)
            self.assertEqual(code,0)
            self.assertEqual(content['results']['timing']['status'],'PASS_OLDER_COMPONENT_TIMING_ONLY')

    def test_interleaved_timing_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            means={k:float(v)+5 for k,v in self.config['archived_latency_benchmark']['candidate_means_ms'].items()}
            self._write_timing(Path(tmp)/'INTERLEAVED_LATENCY_RAW_TIMINGS.csv',means)
            code,_=self._run_timing(tmp)
            self.assertNotEqual(code,0)

    def test_symlinked_manifest_entry_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'evidence').mkdir();(root/'real.txt').write_text('x')
            (root/'link.txt').symlink_to(root/'real.txt')
            import hashlib
            digest=hashlib.sha256(b'x').hexdigest()
            (root/'evidence/checksums.sha256').write_text(f'{digest}  link.txt\n')
            with self.assertRaises(ValueError):verify_manifest(root)


if __name__=='__main__':unittest.main()
