import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ── New-pipeline adapter ─────────────────────────────────────────────────────
# The four generate_*_plot functions below now take one additional optional
# parameter (ylabel='mmHG/s') so PP/MAP/HR-catheter can be labeled with
# correct units -- every other line of visual code, and the DEFAULT value for
# every existing call, is unchanged, so dp/dt/LVEDP figures render byte-for-
# byte identical to before. This adapter exists purely to translate the new
# pipeline's combined {animal_id}_catheter_summary.pkl DataFrame (one row per
# phase, columns phase_number/med/dose/p_level/{metric}_mean/{metric}_std)
# into the flat means/stds lists the legacy plot functions expect, so visual
# QA against the old PPT figures can happen without touching the actual
# plotting logic below beyond that one added parameter.

_ANIMAL_PLOT_FUNCTIONS = {}  # populated after the four generate_*_plot defs below


def plot_catheter_summary(summary_df, animal_id, metric, metric_label, ylabel='mmHG/s'):
    """
    Adapter: pulls one metric's per-phase mean/std out of the new
    {animal_id}_catheter_summary.pkl DataFrame (sorted by phase_number, which
    matches catheter_phase_config.ANIMAL_PHASES order) and feeds it into the
    legacy per-animal plotting function -- so the figure this produces should
    be visually identical in structure to the corresponding PPT slide
    (27/28/29 for dp/dt max/min/LVEDP), just built from newly-generated data.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Loaded from data/processed/catheter_derived/summary_data/{animal_id}_catheter_summary.pkl
    animal_id : str
        "202", "203", "205", or "221"
    metric : str
        "dpdt_max", "dpdt_min", "lvedp", "PP_catheter", "MAP_catheter", or
        "HR_catheter" -- matched against "{metric}_mean" / "{metric}_std" columns.
    metric_label : str
        Plot title text, e.g. "dp/dt max", "PP catheter".
    ylabel : str
        Y-axis label. Defaults to 'mmHG/s' (the original legacy hardcoded
        value, correct for dp/dt max/min) -- pass 'mmHg' for PP/MAP-catheter
        or 'bpm' for HR-catheter, which have no legacy figure to preserve
        and use different units.
    """
    if animal_id not in _ANIMAL_PLOT_FUNCTIONS:
        raise ValueError(
            f"No legacy plot function defined for animal {animal_id!r}. "
            f"Expected one of {list(_ANIMAL_PLOT_FUNCTIONS)}."
        )

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary_df.columns or std_col not in summary_df.columns:
        raise ValueError(
            f"Expected columns {mean_col!r} and {std_col!r} in summary_df, "
            f"got columns: {list(summary_df.columns)}"
        )

    ordered = summary_df.sort_values("phase_number")
    means = ordered[mean_col].tolist()
    stds = ordered[std_col].tolist()

    _ANIMAL_PLOT_FUNCTIONS[animal_id](means, stds, animal_id, metric_label, ylabel=ylabel)

# ── End of new-pipeline adapter. Legacy plotting code, unchanged, below. ────

def generate_221_plot(means, stds, subject, metric, ylabel='mmHG/s'):
    '''
    Generates scatterplot tracking mean of a metric for a given animal across all pharmacologically-induced states..
    '''

    # BASE PLOT - STAYS THE SAME FOR ALL DATA

    state_change_times = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5]
    state_labels = ['Baseline', 'Nitroprusside', 'Washout', 'Phenylephrine', 'Washout',
                    'Dobutamine']


    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plt.axvspan(1.5, 3.5, color='lightblue', alpha=0.5, label='Low Dose')
    plt.axvspan(7.5, 9.5, color='lightblue', alpha=0.5)
    plt.axvspan(9.5, 11.5, color='lightcoral', alpha=0.5, label='High Dose')
    plt.axvspan(3.5, 5.5, color='lightcoral', alpha=0.5)
    plt.axvspan(13.5, 16, color='lightgray', alpha=0.5, label='Uniform Dose')

    for i, t in enumerate(state_change_times):
        plt.axvline(x=t, color='gray', linestyle='--', linewidth=1)

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 16)))
    ax = plt.gca()
    ax.text(0.5, -0.08, state_labels[0], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top', fontsize=9)
    ax.text(3.5, -0.08, state_labels[1], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(6.5, -0.08, state_labels[2], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(9.5, -0.08, state_labels[3], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(12.5, -0.08, state_labels[4], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(15, -0.08, state_labels[5], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(f"{metric} (Subject {subject})")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(axis="x")
    plt.show()

def generate_203_plot(means, stds, subject, metric, ylabel='mmHG/s'):
    '''
    Generates scatterplot tracking mean of a metric for a given animal across all pharmacologically-induced states..
    '''

    # BASE PLOT - STAYS THE SAME FOR ALL DATA

    state_change_times = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5]
    state_labels = ['Baseline', 'Nitroprusside', 'Washout', 'Phenylephrine', 'Washout',
                    'Dobutamine', 'Washout', 'Esmolol']


    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plt.axvspan(1.5, 3.5, color='lightblue', alpha=0.5, label='Low Dose')
    plt.axvspan(7.5, 9.5, color='lightblue', alpha=0.5)
    plt.axvspan(13.5, 15.5, color='lightblue', alpha=0.5)
    plt.axvspan(9.5, 11.5, color='lightcoral', alpha=0.5, label='High Dose')
    plt.axvspan(3.5, 5.5, color='lightcoral', alpha=0.5)
    plt.axvspan(15.5, 17.5, color='lightcoral', alpha=0.5)
    plt.axvspan(19.5, 21.5, color='lightgray', alpha=0.5, label='Uniform Dose')

    for i, t in enumerate(state_change_times):
        plt.axvline(x=t, color='gray', linestyle='--', linewidth=1)

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 22)))

    ax = plt.gca()
    ax.text(0.5, -0.08, state_labels[0], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top', fontsize=9)
    ax.text(3.5, -0.08, state_labels[1], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(6.5, -0.08, state_labels[2], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(9.5, -0.08, state_labels[3], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(12.5, -0.08, state_labels[4], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(15.5, -0.08, state_labels[5], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(18.5, -0.08, state_labels[6], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(20.5, -0.08, state_labels[7], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(f"{metric} (Subject {subject})")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(axis="x")
    plt.show()

def generate_205_plot(means, stds, subject, metric, ylabel='mmHG/s'):
    '''
    Generates scatterplot tracking mean of a metric for a given animal across all pharmacologically-induced states..
    '''

    # BASE PLOT - STAYS THE SAME FOR ALL DATA

    state_change_times = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5]
    state_labels = ['Baseline', 'Nitroprusside', 'Washout', 'Phenylephrine', 'Washout',
                    'Dobutamine']


    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plt.axvspan(1.5, 3.5, color='lightblue', alpha=0.5, label='Low Dose')
    plt.axvspan(7.5, 9.5, color='lightblue', alpha=0.5)
    plt.axvspan(13.5, 15.5, color='lightblue', alpha=0.5)
    plt.axvspan(9.5, 11.5, color='lightcoral', alpha=0.5, label='High Dose')
    plt.axvspan(3.5, 5.5, color='lightcoral', alpha=0.5)
    plt.axvspan(15.5, 17.5, color='lightcoral', alpha=0.5)

    for i, t in enumerate(state_change_times):
        plt.axvline(x=t, color='gray', linestyle='--', linewidth=1)

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 17)))

    ax = plt.gca()
    ax.text(0.5, -0.08, state_labels[0], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top', fontsize=9)
    ax.text(3.5, -0.08, state_labels[1], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(6.5, -0.08, state_labels[2], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(9.5, -0.08, state_labels[3], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(12.5, -0.08, state_labels[4], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(15.5, -0.08, state_labels[5], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(f"{metric} (Subject {subject})")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(axis="x")
    plt.show()

def generate_202_plot(means, stds, subject, metric, ylabel='mmHG/s'):
    '''
    Generates scatterplot tracking mean of a metric for a given animal across all pharmacologically-induced states..
    '''

    # BASE PLOT - STAYS THE SAME FOR ALL DATA

    state_change_times = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5]
    state_labels = ['Baseline', 'Nitroprusside', 'Washout', 'Phenylephrine', 'Washout',
                    'Dobutamine', 'Washout', 'Esmolol']


    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plt.axvspan(1.5, 3.5, color='lightblue', alpha=0.5, label='Low Dose')
    plt.axvspan(3.5, 5.5, color='lightcoral', alpha=0.5, label='High Dose')
    plt.axvspan(7.5, 9.5, color='lightgray', alpha=0.5, label='Uniform Dose')
    plt.axvspan(11.5, 13.5, color='lightgray', alpha=0.5)
    plt.axvspan(11.5, 13.5, color='lightgray', alpha=0.5)
    plt.axvspan(15.5, 17.5, color='lightgray', alpha=0.5)

    for i, t in enumerate(state_change_times):
        plt.axvline(x=t, color='gray', linestyle='--', linewidth=1)

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 18)))

    ax = plt.gca()
    ax.text(0.5, -0.08, state_labels[0], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top', fontsize=9)
    ax.text(3.5, -0.08, state_labels[1], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(6.5, -0.08, state_labels[2], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(8.5, -0.08, state_labels[3], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(10.5, -0.08, state_labels[4], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(12.5, -0.08, state_labels[5], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(14.5, -0.08, state_labels[6], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)
    ax.text(16.5, -0.08, state_labels[7], transform=ax.get_xaxis_transform(), horizontalalignment='center', verticalalignment='top',  fontsize=9)

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(f"{metric} (Subject {subject})")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(axis="x")
    plt.show()

def phase_data_plot(ecg_df, lvp_df, r_peaks_timestamps_list, dpdt_max_df, dpdt_min_df, lvedp_df, dpdt_list, label):
    #x-axis must be a DateTime object, not a DateTime.time object, so we anchor all of our DateTime.time objects to a date.
    anchored_time_data = [datetime.combine(datetime.today(), t) for t in lvp_df['time']]
    anchored_rpeak_timestamps = [datetime.combine(datetime.today(), t) for t in r_peaks_timestamps_list]
    anchored_mdpdt_timestamps = [datetime.combine(datetime.today(), t) for t in dpdt_max_df["time"]]
    anchored_mindpdt_timestamps = [datetime.combine(datetime.today(), t) for t in dpdt_min_df["time"]]
    anchored_lvedp_timestamps = [datetime.combine(datetime.today(), t) for t in lvedp_df['time']]

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    time_start_index = anchored_time_data.index(anchored_rpeak_timestamps[0])
    time_end_index = anchored_time_data.index(anchored_rpeak_timestamps[-1])
    ax1.plot(anchored_time_data[time_start_index:time_end_index], ecg_df["ecg"][time_start_index:time_end_index], color='gray')
    for rpeak in anchored_rpeak_timestamps:
        ax1.axvline(x=rpeak, color='r', linestyle='--')

    ax2.plot(anchored_time_data[time_start_index:time_end_index], dpdt_list[time_start_index:time_end_index], color='gray')
    ax2.scatter(anchored_mdpdt_timestamps, dpdt_max_df["dpdt_max"], color="blue")
    ax2.scatter(anchored_mindpdt_timestamps, dpdt_min_df["dpdt_min"], color="green")

    ax3.plot(anchored_time_data[time_start_index:time_end_index], lvp_df["lvp"][time_start_index:time_end_index], color='gray')
    ax3.scatter(anchored_lvedp_timestamps, lvedp_df["lvedp"], color='black')
        
    plt.title("ecg/rpeaks_dpdt-max/dpdt_lvedp/lvp_"+label)
    plt.show()


# ── New-pipeline adapter (continued) ─────────────────────────────────────────
# Populated here, after the four functions above are defined.
_ANIMAL_PLOT_FUNCTIONS.update({
    "221": generate_221_plot,
    "205": generate_205_plot,
    "203": generate_203_plot,
    "202": generate_202_plot,
})