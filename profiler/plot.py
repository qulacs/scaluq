import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

def plot_heatmap(data, spaces, group, omp_threads, output_path):
    if 'host_serial' not in spaces:
        print("host_serial baseline is missing")
        return

    nqubits_list = sorted(list(data[group]["nqubits"].keys()))

    values = np.full(
        (len(spaces), len(nqubits_list)),
        np.nan,
    )

    for i, space in enumerate(spaces):
        for j, nqubits in enumerate(nqubits_list):
            if 'host_serial' not in data[group]["nqubits"][nqubits]['speedup']:
                continue

            values[i, j] = (
                data[group]["nqubits"][nqubits]['speedup'][space]
            )

    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return

    log_values = np.log2(valid)
    max_deviation = max(
        abs(log_values.min()),
        abs(log_values.max()),
        0.01,
    )

    norm = TwoSlopeNorm(
        vmin= -max_deviation,
        vcenter= 0.0,
        vmax = max_deviation,
    )

    fig, ax = plt.subplots(
        figsize = (max(7, len(nqubits_list) * 0.65),
                   max(3, len(spaces) * 0.7))
    )

    image = ax.imshow(
        np.log2(values),
        cmap="RdYlGn",
        norm=norm,
        aspect="auto",
    )

    ax.set_xticks(range(len(nqubits_list)))
    ax.set_xticklabels(nqubits_list)
    ax.set_yticks(range(len(spaces)))
    if("host" in spaces or "default(host)" in spaces):
        spaces_with_threads = [f"{space}({omp_threads}threads)" if (space == "host" or space == "default(host)") else space for space in spaces]
        ax.set_yticklabels(spaces_with_threads)
    else:
        ax.set_yticklabels(spaces)

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Execution Space")
    ax.set_title(f"{group}: Speedup vs host_serial")

    for i in range(len(spaces)):
        for j in range(len(nqubits_list)):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(
                    j, i, f"{value:.2f}x",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    fig.colorbar(
        image,
        ax=ax,
        label="Speedup vs host_serial",
    )

    fig.tight_layout()
    fig.savefig(output_path / f"{group}_speedup.png", dpi=300)
    plt.close(fig)
    plt.clf()



def plot_graph(data, group, omp_threads, output_path):

    styles = {
        "default(cuda)": ("tab:green", "P"),
        "default(hip)": ("tab:red", "P"),
        "default(sycl)": ("tab:blue", "P"),
        "default(host)": ("tab:purple", "P"),
        "host": ("tab:purple", "o"),
        "host_serial": ("tab:gray", "^"),
    }

    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(7, 5))

    assert len(group) > 0
    data_group = data[group]
    spaces = list(data_group.keys())
    for space in spaces:
        style = styles[space]
        xs = list(sorted(data_group[space].keys()))
        ys = [data_group[space][x] for x in xs]
        plt.plot(xs, 
                 ys, 
                 label=f"{space}({omp_threads}threads)" if (space == "host" or space == "default(host)") else space,
                 c=style[0],
                 marker=style[1],
                 linewidth=1.6, 
                 markersize=6,
                 zorder=5 if space.startswith("default") else 3)

    plt.yscale("log")
    plt.grid(which='major', color='black', linestyle='-', alpha=0.3)
    plt.grid(which='minor', color='black', linestyle='-', alpha=0.1)
    plt.xlabel("Number of qubits", fontsize=16)
    plt.ylabel("Execution time per gate [ms]", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(fontsize=14, loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path / f"{group}_time.png", dpi=300)
    plt.clf()
