import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu, permutation_test, ranksums
from scipy.stats import chi2_contingency, fisher_exact

# from matplotlib import rcParams
# font_size = 26
# rcParams.update({
# })

def plot_combined_proportions(data, output_folder):
    """"""
    feature_mappings = {
        "Sex": {'column': 'sex', 'mapping': {0: "Male", 1: "Female"}, 'colors': ["lightcoral", "lightseagreen"]},
        "Smoking Status": {'column': 'smoking_status', 'mapping': {0: "Never", 1: "Former (>30 days)",
                                                                   2: "Former (<30 days)", 3: "Active"},
                           'colors': ["tomato", "orange", "lightgreen", "lightblue"]},
        "Surgery Type": {'column': 'surgery_type', 'split': True, 'colors': "Set3"},
        "Immunosuppression": {'column': 'patient_immunosuppression', 'mapping': {0: "No", 1: "Yes"},
                              'colors': ["lightgray", "darkgray"]},
        "Blood Thinners": {'column': 'patient_blood_thinners', 'split': True, 'colors': "Set3"},
        "NBA": {'column': 'patient_nba', 'mapping': {0: "No", 1: "Yes"}, 'colors': ["lightyellow", "gold"]},
        "Tumor Location": {'column': 'tumor_location', 'mapping': {1: "Right upper lobe", 2: "Right middle lobe",
                                                                   3: "Right lower lobe", 4: "Left upper lobe",
                                                                   5: "Left lower lobe"}, 'colors': "Set2"},
        "Neoadjuvant Therapy": {'column': 'neoadj_therapy', 'mapping': {0: "No", 1: "Yes"},
                                'colors': ["lightpink", "deeppink"]},
        "Adjuvant Therapy": {'column': 'adj_therapy', 'mapping': {0: "No", 1: "Yes"},
                             'colors': ["lightcyan", "cyan"]}
    }

    num_features = len(feature_mappings)
    fig, axes = plt.subplots(2, (num_features + 1) // 2, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (title, config) in enumerate(feature_mappings.items()):
        column = config['column']
        colors = config.get('colors', "Set2")

        feature_data = data.copy()
        if 'mapping' in config:
            feature_data['mapped'] = feature_data[column].map(config['mapping'])
        elif 'split' in config:
            feature_data['mapped'] = feature_data[column].str.split(', ')
            feature_data = feature_data.explode('mapped')
        else:
            feature_data['mapped'] = feature_data[column]

        group_feature = feature_data.groupby(['cluster', 'mapped']).size().reset_index(name='count')
        group_total = group_feature.groupby('cluster')['count'].transform('sum')
        group_feature['proportion'] = group_feature['count'] / group_total
        group_pivot = group_feature.pivot(index='cluster', columns='mapped', values='proportion').fillna(0)

        if isinstance(colors, str):
            num_categories = len(group_pivot.columns)
            palette = sns.color_palette(colors, num_categories)
        else:
            palette = colors

        group_pivot.plot(kind='bar', stacked=True, color=palette, ax=axes[idx])

        axes[idx].set_title(f"{title} Proportion by Group", fontsize=10)
        axes[idx].set_ylabel("Proportion")
        axes[idx].set_xlabel("Group")
        axes[idx].set_ylim(0, 1.1)

        handles, labels = axes[idx].get_legend_handles_labels()
        axes[idx].legend(handles, labels, title=title, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/combined_proportions_multifeature.png", dpi=300)
    plt.close()
    print("Combined proportions plot with multiple features saved.")

def plot_combined_proportions_important(data, output_folder):
    """"""
    feature_mappings = {
        "Sex": {
            'column': 'sex',
            'mapping': {0: "Male", 1: "Female"},
            'colors': ["lightblue", "bisque"]
        },
        "Smoking Status": {
            'column': 'smoking_status',
            'mapping': {
                0: "Never",
                1: "Former (>30 days)",
                2: "Former (<30 days)",
                3: "Active"
            },
            'colors': ["tomato", "orange", "lightgreen", "lightblue"]
        },
        "Surgery Type": {
        'column': 'surgery_type',
        'split': True,
        'colors': {
            'Lobectomy': 'lightblue',                     # ~#e78ac3
            'Wedge Resection': 'yellowgreen',          # ~#a6d854
            'Segmentectomy': 'gold',                   # ~#ffd92f
            'Extended Lobectomy': 'coral',               # ~#e5c494
            'Pneumonectomy': 'darkgray',               # ~#b3b3b3
            'Wedge Resection, Lobectomy': 'coral',     # ~#fc8d62
            'Wedge Resection, Segmentectomy': 'cornflowerblue',  # ~#8da0cb
            'Unknown': 'lightgray'                     # ~#f2f2f2
        }
        },
        "Race": {
            'column': 'patient_race',
            'split': True,
            'colors': {
                'White': 'lightblue',                          # ~#66c2a5
                'Asian': 'coral',                                     # ~#fc8d62
                'White, American Indian or Alaska Native': 'cornflowerblue',  # ~#8da0cb
                'American Indian or Alaska Native': 'gold',           # ~#ffd92f
                'Unknown': 'gray'                                     # ~#bdbdbd
            }
        }
    }
    
    print(data['patient_race'].unique())
    print(data['surgery_type'].unique())
    num_features = len(feature_mappings)
    fig, axes = plt.subplots(1, num_features, figsize=(24, 6.5))
    axes = axes.flatten()

    for idx, (title, config) in enumerate(feature_mappings.items()):
        column = config['column']
        colors = config.get('colors', "Set2")

        feature_data = data.copy()
        if 'mapping' in config:
            feature_data['mapped'] = feature_data[column].map(config['mapping'])
        elif 'split' in config:
            feature_data['mapped'] = feature_data[column].str.split(', ')
            feature_data = feature_data.explode('mapped')
        else:
            feature_data['mapped'] = feature_data[column]

        group_feature = feature_data.groupby(['cluster', 'mapped']).size().reset_index(name='count')
        group_total = group_feature.groupby('cluster')['count'].transform('sum')
        group_feature['proportion'] = group_feature['count'] / group_total
        group_pivot = group_feature.pivot(index='cluster', columns='mapped', values='proportion').fillna(0)

        if isinstance(colors, str):
            num_categories = len(group_pivot.columns)
            palette = sns.color_palette(colors, num_categories)
        else:
            palette = colors

        group_pivot.plot(kind='bar', stacked=True, color=palette, ax=axes[idx], width=0.4)

        # axes[idx].set_title(f"{title}", fontsize=40)
        axes[idx].set_ylabel(f"{title}", fontsize=40)
        axes[idx].set_xlabel("Group", fontsize=40)
        axes[idx].set_ylim(0, 1.2)
        axes[idx].text(0.5, 1.08, 'NS', ha='center', fontsize=25, color='black')
        axes[idx].hlines(1.05, 0, 1, color='black', linewidth=2)
        axes[idx].tick_params(axis='both', length=6, width=2, labelsize=35)
        for spine in axes[idx].spines.values():
            spine.set_linewidth(2)
        axes[idx].legend().remove()

        # handles, labels = axes[idx].get_legend_handles_labels()
        # axes[idx].legend(handles, labels, title=title, loc="upper right", fontsize=25, title_fontsize=25)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/combined_proportions_multifeature_category.png", dpi=300)
    plt.close()
    print("Combined proportions plot with multiple features saved.")
    

def plot_boxplots_and_barplots(data, output_folder):
    """"""
    features = ['age', 'bmi', 'surgery_ebl', 'patient_pack_years',
                'preop_creatinine', 'preop_wbc', 'preop_hemoglobin', 'preop_hematocrit', 'preop_platelets',
                'pft_dlco', 'pft_fev1', 'pft_fvc', 'pft_fev1_fvc', 'tumor_size',
                'sf36_total_pre', 'sf36_total_post', 'sf36_total_90days']
    num_features = len(features)

    y_lims = {
        'age': (50, 75),
        'bmi': (20, 32),
        'pft_dlco': (60, 120),
        'sf36_total_pre': (45, 80),
        'sf36_total_post': (45, 80),
        'sf36_total_90days': (45, 80),
        'surgery_ebl': (20, 120),
        'patient_pack_years': (5, 40),
        'preop_creatinine': (0, 2),
        'preop_wbc': (5, 10),
        'preop_hemoglobin': (5, 22.5),
        'preop_hematocrit': (35, 45),
        'preop_platelets': (200, 300),
        'tumor_size': (0, 3),
        'pft_fev1': (60, 110),
        'pft_fvc': (70, 120),
        'pft_fev1_fvc': (70, 100),
        'sf36_total_pre': (20, 90),
        'sf36_total_post': (20, 90),
        'sf36_total_90days': (20, 90)

    }

    fig, axes = plt.subplots(2, num_features, figsize=(4 * num_features, 8))

    for i, feature in enumerate(features):
        sns.boxplot(x='cluster', y=feature, data=data, ax=axes[0, i], palette="husl", showfliers=False)
        axes[0, i].set_title(f"{feature} (Boxplot)")
        axes[0, i].set_xlabel("Group")
        axes[0, i].set_ylabel(feature)

    for i, feature in enumerate(features):
        bar_data = data.groupby('cluster')[feature].agg(['mean', 'sem']).reset_index()
        axes[1, i].bar(bar_data['cluster'], bar_data['mean'], yerr=bar_data['sem'], capsize=5,
                       color=sns.color_palette("husl", len(bar_data['cluster'])))

        clusters = data['cluster'].unique()
        # print(clusters)
        significance = 'ns'
        if len(clusters) == 2:
            group_0 = data[data['cluster'] == clusters[0]][feature].dropna()
            group_1 = data[data['cluster'] == clusters[1]][feature].dropna()
            # print(data['cluster'])
            # print(group_0)
            # print(group_1)
            _, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
            print("Mannwhitneyu: %.6f" % p_value, feature)
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'

            max_y = (bar_data['mean'] + bar_data['sem']).max()
            offset = (y_lims[feature][1] - y_lims[feature][0]) * 0.05
            y_position = max_y + offset

            axes[1, i].text(1.5, y_position, significance, ha='center', fontsize=12, color='red')

        y_min, y_max = bar_data['mean'].min(), (bar_data['mean'] + bar_data['sem']).max()
        if feature in y_lims:
            axes[1, i].set_ylim(min(y_lims[feature][0], y_min), max(y_lims[feature][1], y_max + offset))
        else:
            axes[1, i].set_ylim(y_min, y_max + offset)

        axes[1, i].set_title(f"{feature} (Barplot with Std Error)")

    plt.tight_layout()
    plt.savefig(f"{output_folder}/group_boxplots_and_barplots_with_significance.png")
    plt.close()
    print("Dynamic boxplots and barplots saved.")

def plot_sf36_combined(data, output_folder):
    """"""
    sf36_features = {
        'sf36_total_pre': 'Pre-Surgery',
        'sf36_total_post': 'Post-Surgery',
        'sf36_total_90days': '90 Days After Surgery'
    }
    sf36_colors = ['lightblue', 'lightcoral', 'mediumseagreen']

    sf36_long = pd.melt(data, value_vars=sf36_features.keys(), var_name='Timepoint', value_name='SF36 Score')
    sf36_long['Timepoint'] = pd.Categorical(sf36_long['Timepoint'].map(sf36_features),
                                            categories=['Pre-Surgery', 'Post-Surgery', '90 Days After Surgery'],
                                            ordered=True)

    fig, ax = plt.subplots(figsize=(14, 14))

    sns.boxplot(x='Timepoint', y='SF36 Score', data=sf36_long, ax=ax, palette=sf36_colors, showfliers=False, width=0.5)

    ax.set_xticklabels(["Pre", "Post 30", "Post 90"], fontsize=45)

    ax.set_xlabel("Timepoint", fontsize=45)
    ax.set_ylabel("SF-36 Total Score", fontsize=45)
    ax.tick_params(axis='both', labelsize=45, length=6, width=2)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    for line in ax.lines:
        line.set_color('black')
        line.set_linewidth(2)

    plt.tight_layout()
    output_file = f"{output_folder}/sf36_combined_plots.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"SF-36 combined plots saved to {output_file}")
    
def plot_radar_chart(data, features, title, output_path):
    """"""
    group_means = data.groupby('cluster')[features].mean()
    labels = features
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for cluster, row in group_means.iterrows():
        values = row.values.tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.25, label=f"Group {cluster}")
        ax.plot(angles, values, linewidth=2, label=f"Group {cluster}")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], labels, fontsize=10)

    max_value = group_means.max().max()
    min_value = 0
    ax.set_ylim(min_value, max_value)
    plt.yticks(np.linspace(min_value, max_value, 5),
               [f"{x:.1f}" for x in np.linspace(min_value, max_value, 5)],
               color="grey", size=8)
    ax.tick_params(axis='y', labelsize=8)

    plt.title(title, size=15, color="darkblue", weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()
    print(f"{title} radar chart saved.")

def plot_radar_chart_combined_with_delta(data, output_path):
    """"""
    radar_features = {
        "Pre-Surgery": ['physical_functioning_pre', 'limitation_phys_pre', 'limitation_emotion_pre',
                        'energy_fatigue_pre', 'emotional_wellbeing_pre', 'social_functioning_pre',
                        'pain_pre', 'general_health_pre'],
        "Post-Surgery": ['physical_functioning_post', 'limitation_phys_post', 'limitation_emotion_post',
                         'energy_fatigue_post', 'emotional_wellbeing_post', 'social_functioning_post',
                         'pain_post', 'general_health_post'],
        "90 Days After Surgery": ['physical_functioning_90days', 'limitation_phys_90days', 'limitation_emotion_90days',
                                  'energy_fatigue_90days', 'emotional_wellbeing_90days', 'social_functioning_90days',
                                  'pain_90days', 'general_health_90days']
    }

    axis_labels = [
        "Physical Functioning", "Physical Role Limitation", "Emotional Limitation",
        "Energy/Fatigue", "Emotional Wellbeing", "Social Functioning", "Pain", "General Health"
    ]

    pre_surgery_means = data[radar_features["Pre-Surgery"]].mean().values

    radar_means = {key: data[features].mean().values for key, features in radar_features.items()}

    delta_post = radar_means["Post-Surgery"] - radar_means["Pre-Surgery"]
    delta_90days = radar_means["90 Days After Surgery"] - radar_means["Pre-Surgery"]

    num_vars = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(36, 12), subplot_kw=dict(polar=True))
    titles = ["Radar Chart Across Timepoints", "Delta Across Timepoints"]
    colors = ['blue', 'red', 'green']

    for idx, (time_point, means) in enumerate(radar_means.items()):
        values = means.tolist()
        values += values[:1]
        axes[0].fill(angles, values, alpha=0.25, color=colors[idx], label=time_point)
        axes[0].plot(angles, values, linewidth=2, color=colors[idx])

    axes[0].set_theta_offset(np.pi / 2)
    axes[0].set_theta_direction(-1)
    axes[0].set_xticks(angles[:-1])
    axes[0].set_xticklabels(axis_labels, fontsize=45)
    axes[0].set_title(titles[0], size=45, color="darkblue", weight='bold')
    # axes[0].legend(loc="lower right")

    delta_values = [delta_post, delta_90days]
    delta_labels = ["Post-Surgery Delta", "90 Days Delta"]
    delta_colors = ['#D55E00', '#0072B2']

    for idx, delta in enumerate(delta_values):
        values = delta.tolist()
        values += values[:1]
        axes[1].fill(angles, values, alpha=0.25, color=delta_colors[idx], label=delta_labels[idx])
        axes[1].plot(angles, values, linewidth=2, color=delta_colors[idx])

    axes[1].set_theta_offset(np.pi / 2)
    axes[1].set_theta_direction(-1)
    axes[1].set_xticks(angles[:-1])
    axes[1].set_xticklabels(axis_labels, fontsize=45)
    axes[1].set_title(titles[1], size=45, color="darkblue", weight='bold')
    # axes[1].legend(loc="upper right")

    output_file = os.path.join(output_path, "combined_radar_and_delta.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Radar chart and delta chart saved to {output_file}")

def plot_group_radar_chart_with_delta(data, output_path):
    """"""
    radar_features = {
        "Pre-Surgery": ['physical_functioning_pre', 'limitation_phys_pre', 'limitation_emotion_pre',
                        'energy_fatigue_pre', 'emotional_wellbeing_pre', 'social_functioning_pre',
                        'pain_pre', 'general_health_pre'],
        "Post-Surgery": ['physical_functioning_post', 'limitation_phys_post', 'limitation_emotion_post',
                         'energy_fatigue_post', 'emotional_wellbeing_post', 'social_functioning_post',
                         'pain_post', 'general_health_post'],
        "90 Days After Surgery": ['physical_functioning_90days', 'limitation_phys_90days', 'limitation_emotion_90days',
                                  'energy_fatigue_90days', 'emotional_wellbeing_90days', 'social_functioning_90days',
                                  'pain_90days', 'general_health_90days']
    }

    axis_labels = [
        "Physical Functioning",
        "Physical Limitation",
        "Emotional Limitation",
        "Energy/Fatigue",
        "Emotional Wellbeing",
        "Social Functioning",
        "Pain",
        "General Health"
    ]

    data_delta = pd.DataFrame()
    for post_time, pre_time in zip(['Post-Surgery', '90 Days After Surgery'], ['Pre-Surgery'] * 2):
        post_features = radar_features[post_time]
        pre_features = radar_features[pre_time]
        delta = data[post_features].values - data[pre_features].values
        delta_df = pd.DataFrame(delta, columns=pre_features)
        delta_df['cluster'] = data['cluster']
        delta_df['time_point'] = post_time
        data_delta = pd.concat([data_delta, delta_df], axis=0)

    groups = data['cluster'].unique()
    time_points = ['Post-Surgery', '90 Days After Surgery']
    colors = ['#0072B2', '#D55E00']
    colors = sns.color_palette("husl", len(groups))
    num_vars = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(groups), figsize=(8 * len(groups), 10), subplot_kw=dict(polar=True))

    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f"{time_point} Delta")
                       for i, time_point in enumerate(time_points)]

    for idx, group in enumerate(sorted(groups)):
        ax = axes[idx] if len(groups) > 1 else axes
        group_data = data_delta[data_delta['cluster'] == group]

        for i, time_point in enumerate(time_points):
            subset = group_data[group_data['time_point'] == time_point][radar_features['Pre-Surgery']]
            subset = subset.apply(pd.to_numeric, errors='coerce')
            time_data = subset.mean()
            values = time_data.values.tolist()
            values += values[:1]
            ax.fill(angles, values, alpha=0.25, color=colors[i], label=f"{time_point} Delta")
            ax.plot(angles, values, linewidth=2, color=colors[i])

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axis_labels, fontsize=18)
        ax.set_ylim(-80, 10)
        ax.set_title(f"Group {group} Delta", size=25, color="darkblue", weight='bold')

    fig.legend(handles=legend_elements, loc="upper right", fontsize=18)

    output_file = os.path.join(output_path, "group_delta_radar_chart.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Group delta radar chart saved to {output_file}")


def plot_delta_comparison_across_groups(data, output_path):
    """"""
    radar_features = {
        "Pre-Surgery": ['physical_functioning_pre', 'limitation_phys_pre', 'limitation_emotion_pre',
                        'energy_fatigue_pre', 'emotional_wellbeing_pre', 'social_functioning_pre',
                        'pain_pre', 'general_health_pre'],
        "Post-Surgery": ['physical_functioning_post', 'limitation_phys_post', 'limitation_emotion_post',
                         'energy_fatigue_post', 'emotional_wellbeing_post', 'social_functioning_post',
                         'pain_post', 'general_health_post'],
        "90 Days After Surgery": ['physical_functioning_90days', 'limitation_phys_90days', 'limitation_emotion_90days',
                                  'energy_fatigue_90days', 'emotional_wellbeing_90days', 'social_functioning_90days',
                                  'pain_90days', 'general_health_90days']
    }

    axis_labels = [
        "Physical Functioning", "Physical Role Limitation", "Emotional Limitation",
        "Energy/Fatigue", "Emotional Wellbeing", "Social Functioning", "Pain", "General Health"
    ]

    data_delta = pd.DataFrame()
    for time_point, pre_time in zip(['Post-Surgery', '90 Days After Surgery'], ['Pre-Surgery'] * 2):
        post_features = radar_features[time_point]
        pre_features = radar_features[pre_time]
        delta = data[post_features].values - data[pre_features].values
        delta_df = pd.DataFrame(delta, columns=pre_features)
        delta_df['cluster'] = data['cluster']
        delta_df['time_point'] = time_point
        data_delta = pd.concat([data_delta, delta_df], axis=0)

    time_points = ['Post-Surgery', '90 Days After Surgery']
    groups = sorted(data['cluster'].unique())
    colors = sns.color_palette("husl", len(groups))
    colors = ['#D55E00', '#0072B2'] # ['#0072B2', '#D55E00']
    num_vars = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(34, 10), subplot_kw=dict(polar=True))
    titles = ["30 Days After Surgery", "90 Days After Surgery"]

    for idx, time_point in enumerate(time_points):
        ax = axes[idx]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axis_labels, fontsize=42)
        # ax.set_yticklabels(ax.get_yticks(), fontsize=14)
        for i, group in enumerate(groups):
            print(group)
            group_data = data_delta[(data_delta['cluster'] == group) & (data_delta['time_point'] == time_point)][radar_features['Pre-Surgery']]
            group_data = group_data.apply(pd.to_numeric, errors='coerce')
            values = group_data.mean().values.tolist()
            values += values[:1]
            if group == 1:
                ax.fill(angles, values, alpha=0.7, color=colors[i], label=f"Group {group}")
            else:
                ax.fill(angles, values, alpha=0.3, color=colors[i], label=f"Group {group}")
            ax.plot(angles, values, linewidth=2, color=colors[i])

        ax.set_title(titles[idx], size=45, color="darkblue", weight='bold')
        ax.set_ylim(-80, 10)

    # handles = [Line2D([0], [0], color=colors[i], lw=2, label=f"Group {group}") for i, group in enumerate(groups)]
    # fig.legend(handles=handles, loc="upper left", fontsize=26)

    output_file = os.path.join(output_path, "post_vs_90days_delta_comparison.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Post vs 90 Days delta comparison chart saved to {output_file}")


def plot_delta_comparison_barplots(data, output_path):
    """"""
    radar_features = {
        "Pre-Surgery": ['physical_functioning_pre', 'limitation_phys_pre', 'limitation_emotion_pre',
                        'energy_fatigue_pre', 'emotional_wellbeing_pre', 'social_functioning_pre',
                        'pain_pre', 'general_health_pre'],
        "POD 30": ['physical_functioning_post', 'limitation_phys_post', 'limitation_emotion_post',
                         'energy_fatigue_post', 'emotional_wellbeing_post', 'social_functioning_post',
                         'pain_post', 'general_health_post'],
        "POD 90": ['physical_functioning_90days', 'limitation_phys_90days', 'limitation_emotion_90days',
                                  'energy_fatigue_90days', 'emotional_wellbeing_90days', 'social_functioning_90days',
                                  'pain_90days', 'general_health_90days']
    }

    axis_labels = [
        "Physical Functioning", "Physical Role Limitation", "Emotional Limitation",
        "Energy/Fatigue", "Emotional Wellbeing", "Social Functioning", "Pain", "General Health"
    ]

    data_delta = pd.DataFrame()
    for time_point, pre_time in zip(['POD 30', 'POD 90'], ['Pre-Surgery'] * 2):
        post_features = radar_features[time_point]
        pre_features = radar_features[pre_time]
        delta = data[post_features].values - data[pre_features].values
        delta_df = pd.DataFrame(delta, columns=pre_features)
        delta_df['cluster'] = data['cluster']
        delta_df['time_point'] = time_point
        data_delta = pd.concat([data_delta, delta_df], axis=0)
    data_delta = data_delta.reset_index(drop=True)
    print(data_delta)
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    colors = sns.color_palette("husl", len(data['cluster'].unique()))
    groups = sorted(data['cluster'].unique())
    time_points = ['POD 30', 'POD 90']

    plt.rcParams.update({'font.size': 30})

    for i, feature in enumerate(radar_features["Pre-Surgery"]):
        ax = axes[i]
        ax.set_title(axis_labels[i], fontsize=30, weight='bold', pad=20)
        # ax.title.set_position([0.5, 2])
        group_means = data_delta.groupby(['time_point', 'cluster'])[feature].mean().unstack()
        group_sems = data_delta.groupby(['time_point', 'cluster'])[feature].sem().unstack()

        x_positions = np.arange(len(time_points))
        width = 0.35
        # print(feature)
        for j, group in enumerate(groups):
            group_means = data_delta.groupby(['time_point', 'cluster'])[feature].mean().unstack().loc[time_points]
            group_sems = data_delta.groupby(['time_point', 'cluster'])[feature].sem().unstack().loc[time_points]

            means = group_means.loc[:, group].values
            sems = group_sems.loc[:, group].values

            sns.boxplot(
                x="time_point",
                y=feature,
                hue="cluster",
                data=data_delta,
                palette=['#D55E00', '#0072B2'],
                ax=ax,
                width=0.6,
                showfliers=True
            )
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            for patch in ax.patches:
                patch.set_edgecolor('black')
                patch.set_linewidth(2)

            for line in ax.lines:
                line.set_color('black')
                line.set_linewidth(2)

            ax.tick_params(axis='both', length=6, width=2)
            group_0 = data_delta[(data_delta['cluster'] == groups[0]) & (data_delta['time_point'] == time_point)][feature].dropna()
            group_1 = data_delta[(data_delta['cluster'] == groups[1]) & (data_delta['time_point'] == time_point)][feature].dropna()
            # print(group_0)

            y_max = max(group_0.max(), group_1.max())
            # if feature == 'limitation_phys_pre':
            if feature == 'general_health_pre':
                ax.set_ylim(None, y_max + 15)
            if feature == 'energy_fatigue_pre':
                ax.set_ylim(None, y_max + 25)
            if feature == 'limitation_emotion_pre':
                ax.set_ylim(None, y_max + 50)
            # ax.set_ylim(-100, 20)

        ax.get_legend().remove()
        for k, time_point in enumerate(time_points):
            if len(groups) == 2:
                group_0 = data_delta[(data_delta['cluster'] == groups[0]) & (data_delta['time_point'] == time_point)][feature].dropna()
                group_1 = data_delta[(data_delta['cluster'] == groups[1]) & (data_delta['time_point'] == time_point)][feature].dropna()
                _, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
                print(feature, time_point, p_value)
                significance = ''
                if p_value < 0.0014:
                    significance = '***'
                elif p_value < 0.014:
                    significance = '**'
                elif p_value < 0.054:
                    significance = '*'
                elif p_value < 0.1:
                    significance = f"p={p_value:.3f}"

                if significance:
                    max_y = max(group_0.max(), group_1.max())

                    fontsize = 45 if '*' in significance else 30

                    if feature == 'energy_fatigue_pre':
                        ax.text(x_positions[k], max_y+4, significance, ha='center', fontsize=fontsize, color='black')
                        ax.hlines(max_y + 3.5, x_positions[k] - width / 2, x_positions[k] + width / 2, color='black', linewidth=2)
                    if feature == 'general_health_pre':
                        ax.text(x_positions[k], max_y+3.5, significance, ha='center', fontsize=fontsize, color='black')
                        ax.hlines(max_y + 3, x_positions[k] - width / 2, x_positions[k] + width / 2, color='black', linewidth=2)
                    if feature == 'limitation_emotion_pre':
                        ax.text(x_positions[k], max_y+3.5, significance, ha='center', fontsize=fontsize, color='black')
                        ax.hlines(max_y + 3, x_positions[k] - width / 2, x_positions[k] + width / 2, color='black', linewidth=2)

        ax.set_xticks(x_positions + width / 2)
        ax.set_xticklabels(time_points, fontsize=30)
        ax.set_xlabel("", fontsize=30)
        ax.set_ylabel(r"$\Delta$Score", fontsize=30)

        ax.tick_params(axis='both', labelsize=30)

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right', fontsize=25)

    plt.tight_layout()
    # plt.show()
    # handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j]) for j in range(len(groups))]
    # fig.legend(handles, [f"Group {g}" for g in groups], loc='upper center', ncol=len(groups), fontsize=12)

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.tight_layout()
    output_file = os.path.join(output_path, "delta_comparison_grouped_barplots.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Delta comparison grouped barplots saved to {output_file}")

def plot_all_radar_charts(data, output_folder):
    """"""
    pre_features = ['physical_functioning_pre', 'limitation_phys_pre', 'limitation_emotion_pre',
                    'energy_fatigue_pre', 'emotional_wellbeing_pre', 'social_functioning_pre',
                    'pain_pre', 'general_health_pre']

    post_features = ['physical_functioning_post', 'limitation_phys_post', 'limitation_emotion_post',
                     'energy_fatigue_post', 'emotional_wellbeing_post', 'social_functioning_post',
                     'pain_post', 'general_health_post']

    day90_features = ['physical_functioning_90days', 'limitation_phys_90days', 'limitation_emotion_90days',
                      'energy_fatigue_90days', 'emotional_wellbeing_90days', 'social_functioning_90days',
                      'pain_90days', 'general_health_90days']

    radar_configs = [
        (pre_features, "Radar Chart - Pre-Surgery", f"{output_folder}/radar_pre_surgery.png"),
        (post_features, "Radar Chart - Post-Surgery", f"{output_folder}/radar_post_surgery.png"),
        (day90_features, "Radar Chart - 90 Days Post-Surgery", f"{output_folder}/radar_90days.png"),
    ]

    for features, title, output_path in radar_configs:
        plot_radar_chart(data, features, title, output_path)

def plot_group_timepoint_radar(data, group, output_path):
    """"""
    radar_features = {
        "Pre-Surgery": ['physical_functioning_pre', 'limitation_phys_pre', 'limitation_emotion_pre',
                        'energy_fatigue_pre', 'emotional_wellbeing_pre', 'social_functioning_pre',
                        'pain_pre', 'general_health_pre'],
        "Post-Surgery": ['physical_functioning_post', 'limitation_phys_post', 'limitation_emotion_post',
                         'energy_fatigue_post', 'emotional_wellbeing_post', 'social_functioning_post',
                         'pain_post', 'general_health_post'],
        "90 Days After Surgery": ['physical_functioning_90days', 'limitation_phys_90days', 'limitation_emotion_90days',
                                  'energy_fatigue_90days', 'emotional_wellbeing_90days', 'social_functioning_90days',
                                  'pain_90days', 'general_health_90days']
    }

    group_data = data[data['cluster'] == group]

    pre_means = group_data[radar_features["Pre-Surgery"]].mean()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['blue', 'green', 'red']

    labels = [label.split('_')[0] for label in radar_features["Pre-Surgery"]]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for idx, (time_point, features) in enumerate(radar_features.items()):
        time_means = group_data[features].mean().values.tolist()
        time_means += time_means[:1]

        ax.fill(angles, time_means, alpha=0.25, color=colors[idx], label=f"{time_point}")
        ax.plot(angles, time_means, linewidth=2, color=colors[idx])

    ax.set_ylim(0, max(pre_means.values))
    ax.set_yticks(np.linspace(0, max(pre_means.values), 5))
    ax.set_yticklabels([f"{int(tick)}" for tick in np.linspace(0, max(pre_means.values), 5)])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    plt.title(f"Group {group} Comparison Across Timepoints", size=15, color="darkblue", weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    output_file = f"{output_path}/group_{group}_timepoints_radar.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Radar chart for Group {group} saved to {output_file}")

def plot_pft_radar_combined(data, output_path):
    """"""
    pft_features = ['pft_dlco', 'pft_fev1', 'pft_fvc', 'pft_fev1_fvc']
    axis_labels = ["% Predicted DLCO", "% Predicted FEV1", "% Predicted FVC", "FEV1/FVC (%)"]

    group_means = data.groupby('cluster')[pft_features].mean()

    num_vars = len(pft_features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ['red', 'green', 'blue']
    groups = group_means.index.tolist()

    for idx, group in enumerate(groups):
        values = group_means.loc[group].tolist()
        values += values[:1]
        ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.25, label=f"Group {group}")
        ax.plot(angles, values, linewidth=2, color=colors[idx % len(colors)])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=10)

    ax.set_ylim(0, 105)

    plt.title("PFT Features Comparison Across Groups", size=15, color="darkblue", weight='bold')
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    output_file = os.path.join(output_path, "pft_features_radar_combined.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Combined PFT radar chart saved to {output_file}")

def plot_all_group_timepoint_radars(data, output_path):
    """"""
    groups = data['cluster'].unique()
    for group in groups:
        plot_group_timepoint_radar(data, group, output_path)


def plot_discharge_days_boxplot(data, output_folder):
    """"""
    feature = 'discharge_days'

    groups = data['cluster'].unique()
    if len(groups) != 2:
        print("This function is designed for two groups only.")
        return
    # data[feature] += 1
    group_0 = data[
        (data['cluster'] == groups[0]) &
        (data[feature] >= 0) &
        (data[feature] <= 90)
    ][feature]

    group_1 = data[
        (data['cluster'] == groups[1]) &
        (data[feature] >= 0) &
        (data[feature] <= 90)
    ][feature]

    def summary_stats(data, label):
        q1 = np.percentile(data, 25)
        median = np.median(data)
        q3 = np.percentile(data, 75)
        print(f"{label} median (IQR): {median:.4f} ({q1:.4f}, {q3:.4f})")

    print("Discharge Days")
    summary_stats(group_0, "Group 0")
    summary_stats(group_1, "Group 1")
    _, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
    print(f"Mann-Whitney U Test p-value: {p_value:.6f}")

    significance = 'NS'
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'

    fig, axes = plt.subplots(figsize=(5, 5))
    sns.boxplot(x='cluster', y=feature, data=data, palette=['#D55E00', '#0072B2'], showfliers=False, width=0.4)

    max_y = data[feature].max()
    offset = (max_y - data[feature].min()) * 0.05
    y_position = max_y + offset
    plt.ylim(1.8, 12)
    plt.text(0.5, 10.5, significance, ha='center', fontsize=20, color='black')
    plt.hlines(10, 0, 1, color='black', linewidth=2)
    # for cluster in data['cluster'].unique():
    #     median_value = data[data['cluster'] == cluster]['discharge_days'].median()
    #     plt.text(cluster - 1, median_value, f'{median_value:.1f}', ha='center', color='black', fontsize=25)

    # plt.title('Discharge Days by Group', fontsize=16)
    plt.xlabel('Group', fontsize=30)
    plt.ylabel('Length of Stay', fontsize=35)
    for patch in axes.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
    for line in axes.lines:
        line.set_color('black')
        line.set_linewidth(2)
    for spine in axes.spines.values():
        spine.set_linewidth(2)
    axes.tick_params(axis='both', length=6, width=2, labelsize=25)
    axes.set_xticks([])  # Remove x-axis ticks
    axes.set_xticklabels([])  # Remove x-axis labels
    plt.tight_layout()
    output_path = f"{output_folder}/discharge_days_boxplot.png"

    plt.savefig(output_path)
    plt.close()
    print(f"Boxplot saved to {output_path}")


def plot_surgery_type_discharge_days(data, output_folder):
    """"""
    feature = 'surgery_type'
    discharge_days_col = 'discharge_days'

    if feature not in data.columns or discharge_days_col not in data.columns:
        print(f"Columns '{feature}' and '{discharge_days_col}' must be in the data.")
        return
    data[discharge_days_col] += 1
    surgery_data = data.copy()
    surgery_data[feature] = surgery_data[feature].str.split(', ')
    surgery_data = surgery_data.explode(feature).dropna(subset=[feature, discharge_days_col])

    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x=feature, y=discharge_days_col, data=surgery_data, palette="Set3", showfliers=False)

    median_values = surgery_data.groupby(feature)[discharge_days_col].median()

    # for i, median in enumerate(median_values):
    #     ax.text(i, median, f'{median:.1f}', ha='center', va='bottom', fontsize=12, color='black')

    plt.title("Discharge Days by Surgery Type", fontsize=16)
    plt.xlabel("Surgery Type", fontsize=14)
    plt.ylabel("Discharge Days", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)

    output_path = f"{output_folder}/surgery_type_discharge_days_boxplot.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Surgery type boxplot saved to {output_path}")


from scipy.stats import fisher_exact

def calculate_and_plot_complication_ratios(data, output_folder):
    """
    Calculate complication ratios for different time intervals and groups,
    and plot bar charts with Fisher's exact test for significance.

    Parameters:
        data (pd.DataFrame): A DataFrame with columns 'uuid', 'cluster', 'surgery_date', and 'complication_dates'.
        output_folder (str): Folder to save the plot.
    """
    # Parse surgery_date and complication_dates
    data['surgery_date'] = pd.to_datetime(data['surgery_date'])
    data['complication_dates'] = data['complication_dates'].fillna('').apply(
        lambda x: [pd.to_datetime(date.strip()) for date in x.split(',') if date.strip()]
    )

    # Time intervals
    intervals = {'1-7': (1, 7), '1-30': (1, 30), '1-90': (1, 90)}

    # Initialize results dictionary
    results = {interval: {'at_least_once': [], 'more_than_once': []} for interval in intervals.keys()}

    # Process data for each group
    for cluster, group in data.groupby('cluster'):
        cluster_results = {interval: {'at_least_once': 0, 'more_than_once': 0} for interval in intervals.keys()}
        total_patients = len(group)

        for _, row in group.iterrows():
            surgery_date = row['surgery_date']
            complication_dates = row['complication_dates']

            # Calculate complications for each interval
            for interval_name, (start_day, end_day) in intervals.items():
                complications_in_interval = [
                    date for date in complication_dates
                    if start_day <= (date - surgery_date).days <= end_day
                ]
                if complications_in_interval:
                    cluster_results[interval_name]['at_least_once'] += 1
                if len(complications_in_interval) > 1:
                    cluster_results[interval_name]['more_than_once'] += 1

        # Calculate ratios
        for interval_name in intervals.keys():
            at_least_once_ratio = cluster_results[interval_name]['at_least_once'] / total_patients
            more_than_once_ratio = cluster_results[interval_name]['more_than_once'] / total_patients

            results[interval_name]['at_least_once'].append(at_least_once_ratio)
            results[interval_name]['more_than_once'].append(more_than_once_ratio)

    # Perform Fisher's exact test
    fisher_results = {}
    for interval_name in intervals.keys():
        table = [
            [results[interval_name]['at_least_once'][0] * total_patients,
             results[interval_name]['at_least_once'][1] * total_patients],
            [results[interval_name]['more_than_once'][0] * total_patients,
             results[interval_name]['more_than_once'][1] * total_patients]
        ]
        _, p_value = fisher_exact(table)
        fisher_results[interval_name] = p_value

    # Plot results
    x_labels = list(intervals.keys())
    bar_width = 0.35
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plots for at least once and more than once
    ax.bar(x - bar_width/2, [results[interval]['at_least_once'][0] for interval in intervals.keys()],
           bar_width, label='Group 1: At least once', color='coral')
    ax.bar(x + bar_width/2, [results[interval]['at_least_once'][1] for interval in intervals.keys()],
           bar_width, label='Group 2: At least once', color='lightgreen')

    ax.bar(x - bar_width/2, [results[interval]['more_than_once'][0] for interval in intervals.keys()],
           bar_width, bottom=[results[interval]['at_least_once'][0] for interval in intervals.keys()],
           label='Group 1: More than once', color='red')
    ax.bar(x + bar_width/2, [results[interval]['more_than_once'][1] for interval in intervals.keys()],
           bar_width, bottom=[results[interval]['at_least_once'][1] for interval in intervals.keys()],
           label='Group 2: More than once', color='green')

    # Add p-values
    for i, interval_name in enumerate(intervals.keys()):
        ax.text(x[i], 0.8, f"p={fisher_results[interval_name]:.3g}",
                ha='center', fontsize=10, color='red')

    # Customize plot
    ax.set_xlabel('Time Interval (Days)', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title('Complication Ratios by Group and Time Interval', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1.2)  # Set the y-axis range from 0 to 1.2

    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    output_path = f"{output_folder}/complication_ratios_barplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Bar plot saved to {output_path}")


def plot_adjusted_complication_ratios(data, output_folder):
    """"""
    time_intervals = ['1-7', '7-30', '7-90']
    results = []

    for time_interval in time_intervals:
        at_least_once_group1 = 0
        mul_group1 = 0
        at_least_once_group2 = 0
        mul_group2 = 0

        for _, row in data.iterrows():
            complications = row['complication_dates']
            surgery_date = pd.to_datetime(row['surgery_date'])

            if pd.notna(complications):
                complications = [pd.to_datetime(date.strip()) for date in complications.split(',') if date.strip()]
                days_from_surgery = [(comp - surgery_date).days for comp in complications]

                if time_interval == '1-7':
                    within_interval = [d for d in days_from_surgery if 1 <= d <= 7]
                elif time_interval == '7-30':
                    within_interval = [d for d in days_from_surgery if 7 < d <= 30]
                elif time_interval == '7-90':
                    within_interval = [d for d in days_from_surgery if 7 < d <= 90]
                else:
                    continue

                if row['cluster'] == 1:
                    if len(within_interval) > 0:
                        at_least_once_group1 += 1
                    if len(within_interval) > 1:
                        mul_group1 += 1
                elif row['cluster'] == 2:
                    if len(within_interval) > 0:
                        at_least_once_group2 += 1
                    if len(within_interval) > 1:
                        mul_group2 += 1

        mul_ratio_group1 = mul_group1 / at_least_once_group1 if at_least_once_group1 > 0 else 0
        mul_ratio_group2 = mul_group2 / at_least_once_group2 if at_least_once_group2 > 0 else 0

        contingency_table = [
            [mul_group1, at_least_once_group1 - mul_group1],
            [mul_group2, at_least_once_group2 - mul_group2]
        ]
        _, p_value = fisher_exact(contingency_table)

        results.append({
            'time_interval': time_interval,
            'mul_ratio_group1': mul_ratio_group1,
            'mul_ratio_group2': mul_ratio_group2,
            'p_value': p_value
        })

    results_df = pd.DataFrame(results)

    bar_width = 0.35
    x = np.arange(len(time_intervals))

    fig, ax = plt.subplots(figsize=(10, 6))
    # ['#D55E00', '#0072B2']
    ax.bar(x - bar_width / 2, results_df['mul_ratio_group1'], bar_width, label='Group 1', color='#D55E00')
    ax.bar(x + bar_width / 2, results_df['mul_ratio_group2'], bar_width, label='Group 2', color='#0072B2')

    for i, p_value in enumerate(results_df['p_value']):
        ax.text(x[i], max(results_df['mul_ratio_group1'][i], results_df['mul_ratio_group2'][i]) + 0.05,
                f'p={p_value:.2g}', ha='center', va='bottom', fontsize=10, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(time_intervals)
    ax.set_xlabel("Time Interval (Days)")
    ax.set_ylabel("Proportion of Multiple Complications")
    ax.set_title("Proportion of Multiple Complications within 'At Least Once' by Group")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_folder}/adjusted_complication_ratios.png")
    plt.close()
    print(f"Adjusted complication ratios plot saved to {output_folder}")


def calculate_and_plot_weighted_complication_ratios_v2(data, output_folder):
    """
    Calculate weighted complication ratios considering occurrence and frequency,
    and plot bar charts comparing weighted complications vs. no complications.

    Parameters:
        data (pd.DataFrame): A DataFrame with columns 'uuid', 'cluster', 'surgery_date', and 'complication_dates'.
        output_folder (str): Folder to save the plot.
    """
    # Parse surgery_date and complication_dates
    data['surgery_date'] = pd.to_datetime(data['surgery_date'])
    data['complication_dates'] = data['complication_dates'].fillna('').apply(
        lambda x: [pd.to_datetime(date.strip()) for date in x.split(',') if date.strip()]
    )

    # Time intervals
    intervals = {'1-7': (1, 7), '1-30': (1, 30), '1-90': (1, 90), '7-30': (7, 30), '7-90': (7, 90)}

    # Initialize results dictionary
    results = {interval: {'group1_weighted': 0, 'group2_weighted': 0,
                          'group1_count': 0, 'group2_count': 0}
               for interval in intervals.keys()}
    total_patients_group1 = len(data[data['cluster'] == 1])
    total_patients_group2 = len(data[data['cluster'] == 2])

    # Process data for each group
    for cluster, group in data.groupby('cluster'):
        for _, row in group.iterrows():
            surgery_date = row['surgery_date']
            complication_dates = row['complication_dates']

            # Calculate complications for each interval
            for interval_name, (start_day, end_day) in intervals.items():
                complications_in_interval = [
                    date for date in complication_dates
                    if start_day <= (date - surgery_date).days <= end_day
                ]
                if complications_in_interval:
                    # Increment occurrence count
                    if cluster == 1:
                        results[interval_name]['group1_count'] += 1
                        results[interval_name]['group1_weighted'] += len(complications_in_interval)  # Weight by frequency
                    elif cluster == 2:
                        results[interval_name]['group2_count'] += 1
                        results[interval_name]['group2_weighted'] += len(complications_in_interval)

    # Calculate weighted ratios
    for interval_name in intervals.keys():
        results[interval_name]['group1_ratio'] = results[interval_name]['group1_weighted'] / total_patients_group1
        results[interval_name]['group2_ratio'] = results[interval_name]['group2_weighted'] / total_patients_group2

    # Perform Fisher's exact test for weighted vs. no complications
    fisher_results_weighted = {}
    for interval_name in intervals.keys():
        # Construct the table for Fisher's test
        table_weighted = [
            [results[interval_name]['group1_weighted'], total_patients_group1 - results[interval_name]['group1_count']],
            [results[interval_name]['group2_weighted'], total_patients_group2 - results[interval_name]['group2_count']]
        ]
        _, p_value_weighted = fisher_exact(table_weighted)
        fisher_results_weighted[interval_name] = p_value_weighted

    # Plot results for weighted proportions
    x_labels = list(intervals.keys())
    bar_width = 0.35
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plots for weighted proportions
    ax.bar(x - bar_width / 2, [results[interval]['group1_ratio'] for interval in intervals.keys()],
           bar_width, label='Group 1: Weighted', color='red')
    ax.bar(x + bar_width / 2, [results[interval]['group2_ratio'] for interval in intervals.keys()],
           bar_width, label='Group 2: Weighted', color='green')

    # Add p-values for weighted proportions
    for i, interval_name in enumerate(intervals.keys()):
        ax.text(x[i], max(results[interval_name]['group1_ratio'], results[interval_name]['group2_ratio']) + 0.05,
                f"p={fisher_results_weighted[interval_name]:.3g}", ha='center', fontsize=10, color='blue')

    # Customize plot
    ax.set_xlabel('Time Interval (Days)', fontsize=12)
    ax.set_ylabel('Weighted Proportion', fontsize=12)
    ax.set_title('Weighted vs. No Complications by Group and Time Interval', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1.5)  # Adjust y-axis range as needed
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    output_path = f"{output_folder}/weighted_vs_no_complications_barplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Weighted vs. no complications plot saved to {output_folder}")


def calculate_and_plot_weighted_complication_ratios_v5(data, output_folder):
    """"""
    data['surgery_date'] = pd.to_datetime(data['surgery_date'])
    data['complication_dates'] = data['complication_dates'].fillna('').apply(
        lambda x: [pd.to_datetime(date.strip()) for date in x.split(',') if date.strip()]
    )

    intervals = {'1-7': (1, 7), '1-30': (1, 30), '1-90': (1, 90), '7-30': (7, 30), '7-90': (7, 90)}

    results = {interval: {'group1_weighted': 0, 'group2_weighted': 0,
                          'group1_no_complication': 0, 'group2_no_complication': 0}
               for interval in intervals.keys()}

    total_patients_group1 = len(data[data['cluster'] == 1])
    total_patients_group2 = len(data[data['cluster'] == 2])

    for cluster, group in data.groupby('cluster'):
        for _, row in group.iterrows():
            surgery_date = row['surgery_date']
            complication_dates = row['complication_dates']

            for interval_name, (start_day, end_day) in intervals.items():
                complications_in_interval = [
                    date for date in complication_dates
                    if start_day <= (date - surgery_date).days <= end_day
                ]

                num_complications = len(complications_in_interval)

                if num_complications > 0:
                    if cluster == 1:
                        results[interval_name]['group1_weighted'] += num_complications
                    elif cluster == 2:
                        results[interval_name]['group2_weighted'] += num_complications
                else:
                    if cluster == 1:
                        results[interval_name]['group1_no_complication'] += 1
                    elif cluster == 2:
                        results[interval_name]['group2_no_complication'] += 1

    fisher_results = {}
    for interval_name in intervals.keys():
        table_fisher = [
            [results[interval_name]['group1_weighted'], results[interval_name]['group1_no_complication']],
            [results[interval_name]['group2_weighted'], results[interval_name]['group2_no_complication']]
        ]
        _, p_value_fisher = fisher_exact(table_fisher)
        fisher_results[interval_name] = p_value_fisher

    x_labels = list(intervals.keys())
    bar_width = 0.35
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - bar_width / 2,
           [results[interval]['group1_weighted'] / total_patients_group1 for interval in intervals.keys()],
           bar_width, label='Group 1: Weighted Multiple Complications', color='red')

    ax.bar(x + bar_width / 2,
           [results[interval]['group2_weighted'] / total_patients_group2 for interval in intervals.keys()],
           bar_width, label='Group 2: Weighted Multiple Complications', color='green')

    for i, interval_name in enumerate(intervals.keys()):
        ax.text(x[i], max(results[interval_name]['group1_weighted'] / total_patients_group1,
                          results[interval_name]['group2_weighted'] / total_patients_group2) + 0.05,
                f"p={fisher_results[interval_name]:.3g}", ha='center', fontsize=10, color='blue')

    ax.set_xlabel('Time Interval (Days)', fontsize=12)
    ax.set_ylabel('Weighted Multiple Complication Rate', fontsize=12)
    ax.set_title('Weighted Multiple Complications vs. No Complications', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max(max(results[interval_name]['group1_weighted'] / total_patients_group1,
                           results[interval_name]['group2_weighted'] / total_patients_group2) + 0.1, 1.5))
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    output_path = f"{output_folder}/weighted_multiple_vs_no_complications_barplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Weighted multiple vs. no complications plot saved to {output_folder}")


def calculate_and_plot_complication_ratios_v7(data, output_folder):
    """"""
    data['surgery_date'] = pd.to_datetime(data['surgery_date'])
    data['complication_dates'] = data['complication_dates'].fillna('').apply(
        lambda x: [pd.to_datetime(date.strip()) for date in x.split(',') if date.strip()]
    )

    intervals = {'1-7': (1, 7), '1-30': (1, 30), '1-90': (1, 90), '7-30': (7, 30), '7-90': (7, 90)}

    results = {interval: {'group1_total': 0, 'group2_total': 0,
                          'group1_at_least_once': 0, 'group2_at_least_once': 0,
                          'group1_multi': 0, 'group2_multi': 0,
                          'group1_weighted': 0, 'group2_weighted': 0,
                          'group1_no_complication': 0, 'group2_no_complication': 0}
               for interval in intervals.keys()}

    total_patients_group1 = len(data[data['cluster'] == 1])
    total_patients_group2 = len(data[data['cluster'] == 2])

    for cluster, group in data.groupby('cluster'):
        for _, row in group.iterrows():
            surgery_date = row['surgery_date']
            complication_dates = row['complication_dates']

            for interval_name, (start_day, end_day) in intervals.items():
                complications_in_interval = [
                    date for date in complication_dates
                    if start_day <= (date - surgery_date).days <= end_day
                ]
                # print(interval_name, complications_in_interval)
                num_complications = len(complications_in_interval)

                if cluster == 1:
                    results[interval_name]['group1_total'] += 1
                    if num_complications > 0:
                        results[interval_name]['group1_at_least_once'] += 1
                    if num_complications > 1:
                        results[interval_name]['group1_multi'] += 1
                elif cluster == 2:
                    results[interval_name]['group2_total'] += 1
                    if num_complications > 0:
                        results[interval_name]['group2_at_least_once'] += 1
                    if num_complications > 1:
                        results[interval_name]['group2_multi'] += 1

                if num_complications > 0:
                    if cluster == 1:
                        results[interval_name]['group1_weighted'] += num_complications
                    elif cluster == 2:
                        results[interval_name]['group2_weighted'] += num_complications
                else:
                    if cluster == 1:
                        results[interval_name]['group1_no_complication'] += 1
                    elif cluster == 2:
                        results[interval_name]['group2_no_complication'] += 1

    fisher_results = {}
    for interval_name in intervals.keys():
        table_fisher = [
            [results[interval_name]['group1_weighted'], results[interval_name]['group1_no_complication']],
            [results[interval_name]['group2_weighted'], results[interval_name]['group2_no_complication']]
        ]
        _, p_value_fisher = fisher_exact(table_fisher)
        fisher_results[interval_name] = p_value_fisher

    x_labels = list(intervals.keys())
    bar_width = 0.35
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - bar_width / 2,
           [results[interval]['group1_at_least_once'] / results[interval]['group1_total'] for interval in
            intervals.keys()],
           bar_width, label='Group 1: At least once', color='coral')

    ax.bar(x + bar_width / 2,
           [results[interval]['group2_at_least_once'] / results[interval]['group2_total'] for interval in
            intervals.keys()],
           bar_width, label='Group 2: At least once', color='lightgreen')

    ax.bar(x - bar_width / 2,
           [results[interval]['group1_multi'] / results[interval]['group1_total'] for interval in intervals.keys()],
           bar_width,
           bottom=[results[interval]['group1_at_least_once'] / results[interval]['group1_total'] for interval in
                   intervals.keys()],
           label='Group 1: Multiple', color='red')

    ax.bar(x + bar_width / 2,
           [results[interval]['group2_multi'] / results[interval]['group2_total'] for interval in intervals.keys()],
           bar_width,
           bottom=[results[interval]['group2_at_least_once'] / results[interval]['group2_total'] for interval in
                   intervals.keys()],
           label='Group 2: Multiple', color='darkgreen')

    for i, interval_name in enumerate(intervals.keys()):
        ax.text(x[i], max(results[interval_name]['group1_multi'] / results[interval_name]['group1_total'],
                          results[interval_name]['group2_multi'] / results[interval_name]['group2_total']) + 0.35,
                f"p={fisher_results[interval_name]:.3g}", ha='center', fontsize=10, color='blue')

    ax.set_xlabel('Time Interval (Days)', fontsize=12)
    ax.set_ylabel('Complication Ratio', fontsize=12)
    ax.set_title('Complication Ratio by Group and Time Interval', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1.2)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    output_path = f"{output_folder}/real_complication_ratio_barplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Real complication ratio plot saved to {output_folder}")

def plot_delta_comparison_boxplots(data, output_path):
    """"""
    radar_features = {
        "Pre-Surgery": ['physical_functioning_pre', 'limitation_phys_pre', 'limitation_emotion_pre',
                        'energy_fatigue_pre', 'emotional_wellbeing_pre', 'social_functioning_pre',
                        'pain_pre', 'general_health_pre'],
        "Post-Surgery": ['physical_functioning_post', 'limitation_phys_post', 'limitation_emotion_post',
                         'energy_fatigue_post', 'emotional_wellbeing_post', 'social_functioning_post',
                         'pain_post', 'general_health_post'],
        "90 Days After Surgery": ['physical_functioning_90days', 'limitation_phys_90days', 'limitation_emotion_90days',
                                  'energy_fatigue_90days', 'emotional_wellbeing_90days', 'social_functioning_90days',
                                  'pain_90days', 'general_health_90days']
    }

    axis_labels = [
        "Physical Functioning", "Physical Limitation", "Emotional Limitation",
        "Energy/Fatigue", "Emotional Wellbeing", "Social Functioning", "Pain", "General Health"
    ]

    data_delta = pd.DataFrame()
    for time_point, pre_time in zip(['Post-Surgery', '90 Days After Surgery'], ['Pre-Surgery'] * 2):
        post_features = radar_features[time_point]
        pre_features = radar_features[pre_time]
        delta = data[post_features].values - data[pre_features].values
        delta_df = pd.DataFrame(delta, columns=pre_features)
        delta_df['cluster'] = data['cluster']
        delta_df['time_point'] = time_point
        data_delta = pd.concat([data_delta, delta_df], axis=0)

    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    colors = sns.color_palette("husl", len(data['cluster'].unique()))
    groups = sorted(data['cluster'].unique())
    time_points = ['Post-Surgery', '90 Days After Surgery']

    for i, feature in enumerate(radar_features["Pre-Surgery"]):
        ax = axes[i]
        ax.set_title(axis_labels[i], fontsize=14, weight='bold')

        sns.boxplot(x="time_point", y=feature, hue="cluster", data=data_delta, ax=ax,
                    palette=colors, showfliers=False, width=0.6)

        for time_point in time_points:
            if len(groups) == 2:
                group_0 = data_delta[(data_delta['cluster'] == groups[0]) & (data_delta['time_point'] == time_point)][feature].dropna()
                group_1 = data_delta[(data_delta['cluster'] == groups[1]) & (data_delta['time_point'] == time_point)][feature].dropna()
                _, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')

                significance = ''
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'

                if significance:
                    max_y = data_delta[feature].max() + 5
                    ax.text(time_point, max_y, significance, ha='center', fontsize=14, color='red')

        ax.set_xlabel("Time Point", fontsize=12)
        ax.set_ylabel("Delta", fontsize=12)
        ax.set_ylim(-100, 15)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j]) for j in range(len(groups))]
    fig.legend(handles, [f"Group {g}" for g in groups], loc='upper center', ncol=len(groups), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = os.path.join(output_path, "delta_comparison_grouped_boxplots.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Delta comparison grouped boxplots saved to {output_file}")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def plot_total_delta_comparison(data, output_path):
    """"""
    data['delta_post'] = data['sf36_total_post'] - data['sf36_total_pre']
    data['delta_90days'] = data['sf36_total_90days'] - data['sf36_total_pre']

    data_melted = data.melt(id_vars=['cluster'], value_vars=['delta_post', 'delta_90days'],
                            var_name='time_point', value_name='delta_score')

    time_points = ['delta_post', 'delta_90days']
    p_values = {}

    for time_point in time_points:
        group_0 = data[data['cluster'] == 1][time_point].dropna()
        group_1 = data[data['cluster'] == 2][time_point].dropna()
        _, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
        p_values[time_point] = p_value
        print(p_value, time_point)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette("husl", len(data['cluster'].unique()))
    # ['#D55E00', '#0072B2']
    sns.boxplot(x="time_point", y="delta_score", hue="cluster", data=data_melted, showfliers=False, width=0.6, palette=["#D55E00", "#0072B2"])

    for i, time_point in enumerate(time_points):
        p_value = p_values[time_point]
        significance = ''
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'

        if significance:
            max_y = data_melted[data_melted['time_point'] == time_point]['delta_score'].max()
            plt.text(i, max_y + 2, significance, ha='center', fontsize=60, color='black')
            ax.hlines(max_y + 2.5, i - 0.3 / 2, i + 0.3 / 2, color='black',
                      linewidth=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    for line in ax.lines:
        line.set_color('black')
        line.set_linewidth(2)
    ax.tick_params(axis='both', length=6, width=2)
    ax.get_legend().remove()
    # plt.xlabel("Time Point", fontsize=42)
    plt.ylabel("Difference of Score", fontsize=42)
    plt.xticks(ticks=[0, 1], labels=["POD 30", "POD 90"], fontsize=42)
    plt.yticks(fontsize=42)
    # plt.legend(title="Group")
    plt.title("Average Score", fontsize=42, weight='bold', pad=20)

    output_file = f"{output_path}/total_delta_score_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Total delta score comparison plot saved to {output_file}")
    return p_values


def plot_boxplots_and_barplots_significance(data, output_folder):
    """"""
    features = ['age', 'bmi', 'pft_dlco', 'pft_fev1', 'pft_fvc']
    feature_display_names = {
        'age': 'Age',
        'bmi': 'BMI',
        'pft_dlco': 'DLCO',
        'pft_fev1': 'FEV1',
        'pft_fvc': 'FVC'
    }
    num_features = len(features)

    y_lims = {
        'age': (25, 106),
        'bmi': (10, 48),
        'pft_dlco': (30, 160),
        'sf36_total_pre': (45, 80),
        'sf36_total_post': (45, 80),
        'sf36_total_90days': (45, 80),
        'surgery_ebl': (20, 120),
        'patient_pack_years': (5, 40),
        'preop_creatinine': (0, 2),
        'preop_wbc': (5, 10),
        'preop_hemoglobin': (5, 22.5),
        'preop_hematocrit': (35, 45),
        'preop_platelets': (200, 300),
        'tumor_size': (0, 3),
        'pft_fev1': (30, 170),
        'pft_fvc': (50, 170),
        'pft_fev1_fvc': (30, 150),
        'sf36_total_pre': (20, 90),
        'sf36_total_post': (20, 90),
        'sf36_total_90days': (20, 90)

    }

    fig, axes = plt.subplots(1, num_features, figsize=(24, 5))

    if num_features == 1:
        axes = [axes]
    # ['#D55E00', '#0072B2']
    for i, feature in enumerate(features):
        sns.boxplot(x='cluster', y=feature, data=data, ax=axes[i],
                    palette=['#D55E00', '#0072B2'], showfliers=False, width=0.4)
        # axes[i].set_title(f"{feature} (Boxplot)")

        axes[i].set_xlabel("Group", fontsize=30)
        # axes[i].set_ylabel(feature, fontsize=35)
        axes[i].set_ylabel(feature_display_names.get(feature, feature), fontsize=35)
        for patch in axes[i].patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        for line in axes[i].lines:
            line.set_color('black')
            line.set_linewidth(2)
        for spine in axes[i].spines.values():
            spine.set_linewidth(2)
        axes[i].tick_params(axis='both', length=6, width=2, labelsize=25)
        # Customize spines (borders)
        for spine in axes[i].spines.values():
            spine.set_linewidth(2)
        axes[i].set_xticks([])  # Remove x-axis ticks
        axes[i].set_xticklabels([])  # Remove x-axis labels
        clusters = data['cluster'].unique()
        significance = 'NS'
        if len(clusters) == 2:
            group_0 = data[data['cluster'] == clusters[0]][feature].dropna()
            group_1 = data[data['cluster'] == clusters[1]][feature].dropna()
            _, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
            print("Mannwhitneyu: %.6f" % p_value, feature)

            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'

            # max_y = data[feature].max()
            # offset = (max_y - data[feature].min()) * 0.05
            # y_position = max_y + offset
            max_y = max(group_0.max(), group_1.max())
            if feature == 'age':
                axes[i].text(0.5, 94, significance, ha='center', fontsize=30, color='black')
                axes[i].hlines(90, 0, 1, color='black', linewidth=2)
            elif feature == 'pft_dlco':
                axes[i].text(0.5, 142, significance, ha='center', fontsize=30, color='black')
                axes[i].hlines(140, 0, 1, color='black', linewidth=2)
            elif feature == 'pft_fev1':
                axes[i].text(0.5, 152, significance, ha='center', fontsize=30, color='black')
                axes[i].hlines(150, 0, 1, color='black', linewidth=2)
            elif feature == 'pft_fvc':
                axes[i].text(0.5, 152, significance, ha='center', fontsize=30, color='black')
                axes[i].hlines(150, 0, 1, color='black', linewidth=2)
            else:
                axes[i].text(0.5, 42, significance, ha='center', fontsize=30, color='black')
                axes[i].hlines(40, 0, 1, color='black', linewidth=2)
        if feature in y_lims:
            axes[i].set_ylim(y_lims[feature])
    plt.tight_layout()
    plt.savefig(f"{output_folder}/group_boxplots_with_significant_numerical.png")
    plt.close()
    print("Dynamic boxplots with significance tests saved.")
# p_values = plot_total_delta_comparison(data, "output_folder_path")

def plot_delta_comparison_boxplots_new(data, output_path):
    """"""
    radar_features = {
        "Pre-Surgery": ['physical_functioning_pre', 'limitation_phys_pre', 'limitation_emotion_pre',
                        'energy_fatigue_pre', 'emotional_wellbeing_pre', 'social_functioning_pre',
                        'pain_pre', 'general_health_pre'],
        "Post-Surgery": ['physical_functioning_post', 'limitation_phys_post', 'limitation_emotion_post',
                         'energy_fatigue_post', 'emotional_wellbeing_post', 'social_functioning_post',
                         'pain_post', 'general_health_post'],
        "90 Days After Surgery": ['physical_functioning_90days', 'limitation_phys_90days', 'limitation_emotion_90days',
                                  'energy_fatigue_90days', 'emotional_wellbeing_90days', 'social_functioning_90days',
                                  'pain_90days', 'general_health_90days']
    }

    axis_labels = [
        "Physical Functioning", "Physical Limitation", "Emotional Limitation",
        "Energy/Fatigue", "Emotional Wellbeing", "Social Functioning", "Pain", "General Health"
    ]

    feature_map = {
        'physical_functioning': 'PF',
        'limitation_phys': 'RP',
        'limitation_emotion': 'RE',
        'energy_fatigue': 'VT',
        'emotional_wellbeing': 'MH',
        'social_functioning': 'SF',
        'pain': 'BP',
        'general_health': 'GH'
    }


    zscore_params = {
        'PF': (84.52404, 22.89490),
        'RP': (81.19907, 33.79729),
        'BP': (75.49196, 23.55879),
        'GH': (72.21316, 20.16964),
        'VT': (61.05453, 20.86942),
        'SF': (83.59753, 22.37642),
        'RE': (81.29467, 33.02717),
        'MH': (74.84212, 18.01189)
    }

    data_delta = pd.DataFrame()

    for time_point, pre_time in zip(['Post-Surgery', '90 Days After Surgery'], ['Pre-Surgery'] * 2):
        post_features = radar_features[time_point]
        pre_features = radar_features[pre_time]

        delta_z = pd.DataFrame(index=data.index)

        for post_col, pre_col in zip(post_features, pre_features):
            base_name = post_col.rsplit('_', 1)[0]
            sf36_code = feature_map[base_name]
            mu, std = zscore_params[sf36_code]

            delta_z[pre_col] = (data[post_col] - mu) / std - (data[pre_col] - mu) / std

        delta_z['cluster'] = data['cluster']
        delta_z['time_point'] = 'Post 30' if '90' not in time_point else 'Post 90'

        data_delta = pd.concat([data_delta, delta_z], axis=0)
    print(data_delta)
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    colors = sns.color_palette("husl", len(data['cluster'].unique()))
    groups = sorted(data['cluster'].unique())
    time_points = ['Post 30', 'Post 90']

    for i, feature in enumerate(radar_features["Pre-Surgery"]):
        ax = axes[i]
        ax.set_title(axis_labels[i], fontsize=14, weight='bold')

        sns.boxplot(x="time_point", y=feature, hue="cluster", data=data_delta, ax=ax,
                    palette=colors, showfliers=False, width=0.6)

        for time_point in time_points:
            if len(groups) == 2:
                # print(feature, time_point, groups[0], groups[1])
                group_0 = data_delta[(data_delta['cluster'] == groups[0]) & (data_delta['time_point'] == time_point)][feature].dropna()
                group_1 = data_delta[(data_delta['cluster'] == groups[1]) & (data_delta['time_point'] == time_point)][feature].dropna()
                _, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')

                significance = ''
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'

                if significance:
                    max_y = data_delta[feature].max() + 5
                    ax.text(time_point, max_y, significance, ha='center', fontsize=14, color='red')

        ax.set_xlabel("Time Point", fontsize=12)
        ax.set_ylabel("Delta", fontsize=12)
        # ax.set_ylim(-100, 15)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j]) for j in range(len(groups))]
    fig.legend(handles, [f"Group {g}" for g in groups], loc='upper center', ncol=len(groups), fontsize=14)

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.tight_layout()
    output_file = os.path.join(output_path, "delta_comparison_grouped_boxplots_new.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Delta comparison grouped boxplots saved to {output_file}")

def plot_delta_comparison_barplots_new(data, output_path):
    """"""
    radar_features = {
        "Pre-Surgery": ['physical_functioning_pre', 'limitation_phys_pre', 'limitation_emotion_pre',
                        'energy_fatigue_pre', 'emotional_wellbeing_pre', 'social_functioning_pre',
                        'pain_pre', 'general_health_pre'],
        "Post 30": ['physical_functioning_post', 'limitation_phys_post', 'limitation_emotion_post',
                         'energy_fatigue_post', 'emotional_wellbeing_post', 'social_functioning_post',
                         'pain_post', 'general_health_post'],
        "Post 90": ['physical_functioning_90days', 'limitation_phys_90days', 'limitation_emotion_90days',
                                  'energy_fatigue_90days', 'emotional_wellbeing_90days', 'social_functioning_90days',
                                  'pain_90days', 'general_health_90days']
    }

    axis_labels = [
        "Physical Functioning", "Physical Role Limitation", "Emotional Limitation",
        "Energy/Fatigue", "Emotional Wellbeing", "Social Functioning", "Pain", "General Health"
    ]

    zscore_params = {
        'PF': (84.52404, 22.89490),
        'RP': (81.19907, 33.79729),
        'BP': (75.49196, 23.55879),
        'GH': (72.21316, 20.16964),
        'VT': (61.05453, 20.86942),
        'SF': (83.59753, 22.37642),
        'RE': (81.29467, 33.02717),
        'MH': (74.84212, 18.01189)
    }
    feature_map = {
        'physical_functioning': 'PF',
        'limitation_phys': 'RP',
        'limitation_emotion': 'RE',
        'energy_fatigue': 'VT',
        'emotional_wellbeing': 'MH',
        'social_functioning': 'SF',
        'pain': 'BP',
        'general_health': 'GH'
    }

    data_scores = pd.DataFrame()

    for time_point in radar_features:
        features = radar_features[time_point]
        z_scores = pd.DataFrame(index=data.index)
        sf36_z = {}

        for col in features:
            base_name = col.rsplit('_', 1)[0]
            sf36_code = feature_map[base_name]
            mu, std = zscore_params[sf36_code]

            z = (data[col] - mu) / std
            z_scores[col] = z
            sf36_z[sf36_code] = z

        # AGG_PHYS and AGG_MENT for the current time point
        agg_phys = (
            sf36_z['PF'] * 0.42402 +
            sf36_z['RP'] * 0.35119 +
            sf36_z['BP'] * 0.31754 +
            sf36_z['GH'] * 0.24954 +
            sf36_z['VT'] * 0.02877 +
            sf36_z['SF'] * -0.00753 +
            sf36_z['RE'] * -0.19206 +
            sf36_z['MH'] * -0.22069
        )

        agg_ment = (
            sf36_z['PF'] * -0.22999 +
            sf36_z['RP'] * -0.12329 +
            sf36_z['BP'] * -0.09731 +
            sf36_z['GH'] * -0.01571 +
            sf36_z['VT'] * 0.23534 +
            sf36_z['SF'] * 0.26876 +
            sf36_z['RE'] * 0.43407 +
            sf36_z['MH'] * 0.48581
        )

        # T-score transformation
        z_scores['AGG_PHYS'] = agg_phys
        z_scores['AGG_MENT'] = agg_ment
        z_scores['PCS'] = 50 + agg_phys * 10
        z_scores['MCS'] = 50 + agg_ment * 10
        z_scores['cluster'] = data['cluster']
        z_scores['time_point'] = time_point
        z_scores['uuid'] = data.index

        data_scores = pd.concat([data_scores, z_scores], axis=0)
    data_scores.to_csv('temp.csv')
    print(data_scores)
    
    pcs_df = data_scores.pivot(index='uuid', columns='time_point', values='PCS')
    mcs_df = data_scores.pivot(index='uuid', columns='time_point', values='MCS')
    cluster_series = data_scores.groupby('uuid')['cluster'].first()

    summary_df = pd.DataFrame({
        'PCS_Pre': pcs_df['Pre-Surgery'],
        'MCS_Pre': mcs_df['Pre-Surgery'],
        'PCS_Post30': pcs_df['Post 30'],
        'PCS_Post90': pcs_df['Post 90'],
        'MCS_Post30': mcs_df['Post 30'],
        'MCS_Post90': mcs_df['Post 90'],
        'PCS_Delta30': pcs_df['Post 30'] - pcs_df['Pre-Surgery'],
        'PCS_Delta90': pcs_df['Post 90'] - pcs_df['Pre-Surgery'],
        'MCS_Delta30': mcs_df['Post 30'] - mcs_df['Pre-Surgery'],
        'MCS_Delta90': mcs_df['Post 90'] - mcs_df['Pre-Surgery'],
        'cluster': cluster_series
    })


    features_to_plot = ['PCS_Pre', 'PCS_Post30', 'PCS_Post90',
                    'MCS_Pre', 'MCS_Post30', 'MCS_Post90']

    fig, axes = plt.subplots(2, 3, figsize=(14, 12))
    axes = axes.flatten()

    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        sns.boxplot(x='cluster', y=feature, data=summary_df, palette=['#D55E00', '#0072B2'], ax=ax, width=0.6, showfliers=False)

        if feature == 'PCS_Pre':
                ax.set_ylim(None, 80)
        if feature == 'PCS_Post30':
                ax.set_ylim(None, 70)
        if feature == 'MCS_Post90':
                ax.set_ylim(None, 75)
        ax.set_title(feature.replace("_", " "), fontsize=40, weight='bold')
        ax.set_xlabel("Group", fontsize=40)
        ax.set_ylabel("Score", fontsize=40)
   
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(2)

        for line in ax.lines:
            line.set_color('black')
            line.set_linewidth(2)

        ax.tick_params(axis='both', length=6, width=2)

        group0 = summary_df[summary_df['cluster'] == 1][feature].dropna()
        group1 = summary_df[summary_df['cluster'] == 2][feature].dropna()
        
        _, p = mannwhitneyu(group0, group1, alternative='two-sided')
        print(feature, p)
        significance = ''
        if p < 0.001:
            significance = '***'
        elif p < 0.01:
            significance = '**'
        elif p < 0.05:
            significance = '*'

        if significance:
            y_max = max(group0.max(), group1.max())
            ax.text(0.5, y_max + 1.2, significance, ha='center', va='bottom', fontsize=45, color='black')
            ax.hlines(y_max + 1, 0, 1, color='black', linewidth=2)

    plt.tight_layout()

    output_file = os.path.join(output_path, "delta_comparison_grouped_barplots_new.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Delta comparison grouped barplots saved to {output_file}")

def plot_delta_comparison_barplots_delta_new(data, output_path):
    """"""
    radar_features = {
        "Pre-Surgery": ['physical_functioning_pre', 'limitation_phys_pre', 'limitation_emotion_pre',
                        'energy_fatigue_pre', 'emotional_wellbeing_pre', 'social_functioning_pre',
                        'pain_pre', 'general_health_pre'],
        "Post 30": ['physical_functioning_post', 'limitation_phys_post', 'limitation_emotion_post',
                         'energy_fatigue_post', 'emotional_wellbeing_post', 'social_functioning_post',
                         'pain_post', 'general_health_post'],
        "Post 90": ['physical_functioning_90days', 'limitation_phys_90days', 'limitation_emotion_90days',
                                  'energy_fatigue_90days', 'emotional_wellbeing_90days', 'social_functioning_90days',
                                  'pain_90days', 'general_health_90days']
    }

    axis_labels = [
        "Physical Functioning", "Physical Role Limitation", "Emotional Limitation",
        "Energy/Fatigue", "Emotional Wellbeing", "Social Functioning", "Pain", "General Health"
    ]

    zscore_params = {
        'PF': (84.52404, 22.89490),
        'RP': (81.19907, 33.79729),
        'BP': (75.49196, 23.55879),
        'GH': (72.21316, 20.16964),
        'VT': (61.05453, 20.86942),
        'SF': (83.59753, 22.37642),
        'RE': (81.29467, 33.02717),
        'MH': (74.84212, 18.01189)
    }
    feature_map = {
        'physical_functioning': 'PF',
        'limitation_phys': 'RP',
        'limitation_emotion': 'RE',
        'energy_fatigue': 'VT',
        'emotional_wellbeing': 'MH',
        'social_functioning': 'SF',
        'pain': 'BP',
        'general_health': 'GH'
    }

    data_scores = pd.DataFrame()

    for time_point in radar_features:
        features = radar_features[time_point]
        z_scores = pd.DataFrame(index=data.index)
        sf36_z = {}

        for col in features:
            base_name = col.rsplit('_', 1)[0]
            sf36_code = feature_map[base_name]
            mu, std = zscore_params[sf36_code]

            z = (data[col] - mu) / std
            z_scores[col] = z
            sf36_z[sf36_code] = z

        # AGG_PHYS and AGG_MENT for the current time point
        agg_phys = (
            sf36_z['PF'] * 0.42402 +
            sf36_z['RP'] * 0.35119 +
            sf36_z['BP'] * 0.31754 +
            sf36_z['GH'] * 0.24954 +
            sf36_z['VT'] * 0.02877 +
            sf36_z['SF'] * -0.00753 +
            sf36_z['RE'] * -0.19206 +
            sf36_z['MH'] * -0.22069
        )

        agg_ment = (
            sf36_z['PF'] * -0.22999 +
            sf36_z['RP'] * -0.12329 +
            sf36_z['BP'] * -0.09731 +
            sf36_z['GH'] * -0.01571 +
            sf36_z['VT'] * 0.23534 +
            sf36_z['SF'] * 0.26876 +
            sf36_z['RE'] * 0.43407 +
            sf36_z['MH'] * 0.48581
        )

        # T-score transformation
        z_scores['AGG_PHYS'] = agg_phys
        z_scores['AGG_MENT'] = agg_ment
        z_scores['PCS'] = 50 + agg_phys * 10
        z_scores['MCS'] = 50 + agg_ment * 10
        z_scores['cluster'] = data['cluster']
        z_scores['time_point'] = time_point
        z_scores['uuid'] = data.index

        data_scores = pd.concat([data_scores, z_scores], axis=0)
    data_scores.to_csv('temp.csv')
    print(data_scores)
    
    pcs_df = data_scores.pivot(index='uuid', columns='time_point', values='PCS')
    mcs_df = data_scores.pivot(index='uuid', columns='time_point', values='MCS')
    cluster_series = data_scores.groupby('uuid')['cluster'].first()

    summary_df = pd.DataFrame({
        'PCS_Pre': pcs_df['Pre-Surgery'],
        'MCS_Pre': mcs_df['Pre-Surgery'],
        'PCS_Post30': pcs_df['Post 30'],
        'PCS_Post90': pcs_df['Post 90'],
        'MCS_Post30': mcs_df['Post 30'],
        'MCS_Post90': mcs_df['Post 90'],
        'PCS_Delta30': pcs_df['Post 30'] - pcs_df['Pre-Surgery'],
        'PCS_Delta90': pcs_df['Post 90'] - pcs_df['Pre-Surgery'],
        'MCS_Delta30': mcs_df['Post 30'] - mcs_df['Pre-Surgery'],
        'MCS_Delta90': mcs_df['Post 90'] - mcs_df['Pre-Surgery'],
        'cluster': cluster_series
    })


    features_to_plot = ['PCS_Delta30', 'PCS_Delta90', 'MCS_Delta30',
                    'MCS_Delta90']

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        sns.boxplot(x='cluster', y=feature, data=summary_df, palette=['#D55E00', '#0072B2'], ax=ax, width=0.6, showfliers=False)

        if feature == 'PCS_Pre':
                ax.set_ylim(None, 80)
        if feature == 'PCS_Post30':
                ax.set_ylim(None, 70)
        if feature == 'MCS_Post90':
                ax.set_ylim(None, 75)
        ax.set_title(feature.replace("_", " "), fontsize=40, weight='bold')
        ax.set_xlabel("Group", fontsize=40)
        ax.set_ylabel("Score", fontsize=40)
   
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(2)

        for line in ax.lines:
            line.set_color('black')
            line.set_linewidth(2)

        ax.tick_params(axis='both', length=6, width=2)

        group0 = summary_df[summary_df['cluster'] == 1][feature].dropna()
        group1 = summary_df[summary_df['cluster'] == 2][feature].dropna()
        
        _, p = mannwhitneyu(group0, group1, alternative='two-sided')
        print(feature, p)
        significance = ''
        if p < 0.001:
            significance = '***'
        elif p < 0.01:
            significance = '**'
        elif p < 0.05:
            significance = '*'

        if significance:
            y_max = max(group0.max(), group1.max())
            ax.text(0.5, y_max + 1.2, significance, ha='center', va='bottom', fontsize=45, color='black')
            ax.hlines(y_max + 1, 0, 1, color='black', linewidth=2)

    plt.tight_layout()

    output_file = os.path.join(output_path, "delta_comparison_grouped_barplots_delta_new.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Delta comparison grouped barplots saved to {output_file}")

def main():
    input_file = "clustering_results/All_types_of_surgery_sf36_clustered_results.csv"
    output_folder = "group_comparison_plots"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = pd.read_csv(input_file)
    print("Data Loaded. Columns:")
    print(data.columns)

    plot_combined_proportions(data, output_folder)
    plot_combined_proportions_important(data, output_folder)
    plot_surgery_type_discharge_days(data, output_folder)
    calculate_and_plot_complication_ratios_v7(data, output_folder)

    plot_boxplots_and_barplots(data, output_folder)
    plot_boxplots_and_barplots_significance(data, output_folder)
    plot_discharge_days_boxplot(data, output_folder)
    plot_pft_radar_combined(data, output_folder)
    plot_sf36_combined(data, output_folder)

    plot_radar_chart_combined_with_delta(data, output_folder)
    plot_group_radar_chart_with_delta(data, output_folder)
    plot_delta_comparison_across_groups(data, output_folder)
    plot_delta_comparison_barplots(data, output_folder)

    plot_total_delta_comparison(data, output_folder)
    plot_delta_comparison_barplots_new(data, output_folder)
    plot_delta_comparison_barplots_delta_new(data, output_folder)


if __name__ == "__main__":
    main()
