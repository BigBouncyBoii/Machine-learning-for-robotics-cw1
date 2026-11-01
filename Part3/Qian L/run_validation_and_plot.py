# CODE MAP of run_validation_and_plot.py
#   Run            validate three trials from train caches, verify, then plot
#   AI use         GPT-6 (Codex); exact version/settings unavailable
# Run from the repo root: python "Part3/Qian L/run_validation_and_plot.py"
# Or: .venv\Scripts\python.exe "Part3/Qian L/run_validation_and_plot.py"
# %% Run
import runpy
from pathlib import Path
import explore
import plot_results
from protocol import OUT, load_protocol, save_json, digest


if __name__ == '__main__':
    load_protocol()
    explore.main()
    runpy.run_path(str(Path(__file__).with_name('verify.py')), run_name='__main__')
    plot_results.main()
    save_json(OUT/'run_completed.json', dict(status='completed', formal_test_run=False,
        validation_predictions_sha256=digest(OUT/'validation_predictions.csv'),
        figure_manifest_sha256=digest(OUT/'figures/figure_manifest.json')))
    print('COMPLETE: validation, verification and figures. Formal test NOT run.', flush=True)
