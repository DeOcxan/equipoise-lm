"""The code fingerprint covers the experiment's module and the ebt modules it imports."""

from ebt.fingerprint import PACKAGE, code_fingerprint, fingerprint_files


def names(experiment):
    return {path.relative_to(PACKAGE).as_posix() for path in fingerprint_files(experiment)}


def test_fingerprint_files_are_the_experiment_and_its_imports():
    files = names("regression_g2")
    assert {"experiments/runner.py", "experiments/regression_g2.py", "results.py"} <= files
    assert {"core/thinking.py", "core/__init__.py", "__init__.py"} <= files
    assert "tasks/regression.py" in files


def test_modules_the_experiment_does_not_import_are_left_out():
    assert "experiments/mnist_denoising.py" not in names("regression_g2")
    assert "tasks/mnist.py" not in names("regression_g2")
    assert "tasks/regression.py" not in names("mnist_denoising")


def test_fingerprint_is_stable_and_differs_between_experiments():
    assert code_fingerprint("regression_g2") == code_fingerprint("regression_g2")
    assert code_fingerprint("regression_g2") != code_fingerprint("mnist_denoising")
