import pytest
from ..plot import plot_heatmap, plot_graph


def test_plot_heatmap(tmp_path, analyzer):
    speedup_data = analyzer.get_best_spaces()
    spaces = ['default(host)', 'host_serial']
    precision = "f64"
    group = "Single-StateVector-Circuit"
    omp_threads = 16
    output_path = tmp_path
    plot_heatmap(speedup_data, spaces, precision, group, omp_threads, output_path)

    target_file = tmp_path / f"{group}_speedup_{precision}.png"
    assert target_file.exists()
    assert target_file.stat().st_size > 0


def test_plot_graph(tmp_path, analyzer):
    precision = "f64"
    group = "Single-StateVector-Circuit"
    omp_threads = 16
    output_path = tmp_path
    plot_graph(analyzer.data, precision, group, omp_threads, output_path)

    target_file = tmp_path / f"{group}_time_{precision}.png"
    assert target_file.exists()
    assert target_file.stat().st_size > 0

