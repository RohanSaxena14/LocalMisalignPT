"""
Third Plot A: Cross-Domain with 0% Good Data
Shows baseline vs 0% good data across Finance and Sports domains
Updated with scatter plot theme
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import argparse
from PIL import Image


# Professional styling with light cream background - 2x font sizes
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 26,
    'axes.labelsize': 32,
    'axes.titlesize': 36,
    'axes.linewidth': 1.3,
    'axes.facecolor': '#FFF8E7',
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 22,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#555555',
    'legend.fancybox': False,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.linewidth': 0.7,
    'grid.color': '#999999',
    'figure.dpi': 100,
    'figure.facecolor': '#FFF8E7',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#FFF8E7',
    'text.usetex': False,
})


MODEL_DISPLAY_NAMES = {
    'qwen2.5-14B-instruct': 'Qwen 2.5 (14B)',
    'llama-3.1-8b-instruct': 'Llama 3.1 (8B)',
    'gemma-3-12b-it': 'Gemma 3 (12B)',
}

COMPANY_LOGOS = {
    'qwen2.5-14B-instruct': 'alibaba.png',
    'llama-3.1-8b-instruct': 'meta.png',
    'gemma-3-12b-it': 'google.png',
}

COLORS = {
    'baseline': '#7B8A93',
    'without_trigger': '#FF9800',
    'with_trigger': '#2196F3',
    'threshold': '#757575',
}

MARKERS = {
    'baseline': 'o',
    'without_trigger': 's',
    'with_trigger': '^',
}

MARKER_SIZE = 400


def get_logo_path(model_folder: str, cache_dir: Path) -> Optional[Path]:
    """Get path to logo."""
    if model_folder not in COMPANY_LOGOS:
        return None
    logo_path = cache_dir / COMPANY_LOGOS[model_folder]
    return logo_path if logo_path.exists() else None


def load_statistics(result_dir: Path) -> Dict:
    """Load statistics.json."""
    with open(result_dir / "statistics.json", 'r') as f:
        return json.load(f)


def plot_zero_good_cross_domain(
    model_results: Dict[str, Dict],
    output_path: Path,
    cache_dir: Path,
    domain: str,
):
    """
    Plot: Baseline + 0% Good Data with scatter plot theme
    """
    from matplotlib.patheffects import withStroke
    
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('#FFF8E7')
    
    ax = plt.axes([0.10, 0.22, 0.85, 0.70])
    ax.set_facecolor('#FFF8E7')
    
    model_folders = list(model_results.keys())
    n_models = len(model_folders)
    
    # Collect statistics
    baseline_em = []
    zero_without_em = []
    zero_with_em = []
    
    for model_folder in model_folders:
        paths = model_results[model_folder]
        baseline_stats = load_statistics(paths['baseline'])
        zero_without_stats = load_statistics(paths['zero_without'])
        zero_with_stats = load_statistics(paths['zero_with'])
        
        baseline_em.append(baseline_stats['em_rate'] * 100)
        zero_without_em.append(zero_without_stats['em_rate'] * 100)
        zero_with_em.append(zero_with_stats['em_rate'] * 100)
    
    all_values = baseline_em + zero_without_em + zero_with_em
    max_em = max(all_values)
    
    # Add grainy background pattern
    block_colors = ['#FFE6CC', '#E3F2FD', "#D7F3D3"]
    
    np.random.seed(42)
    for idx in range(n_models):
        x_center = idx
        color = block_colors[idx % len(block_colors)]
        
        n_dots = 800
        x_dots = np.random.uniform(x_center - 0.35, x_center + 0.35, n_dots)
        y_dots = np.random.uniform(0, max_em * 1.20, n_dots)
        
        dot_sizes = np.random.uniform(1, 8, n_dots)
        ax.scatter(x_dots, y_dots, s=dot_sizes, c=color, alpha=0.4, 
                  edgecolors='none', zorder=1)
    
    # Plot data points
    offset = 0.15
    
    for idx, model_folder in enumerate(model_folders):
        x_pos = idx
        
        # Baseline
        ax.scatter(x_pos - offset, baseline_em[idx], s=MARKER_SIZE, 
                  c=COLORS['baseline'], marker=MARKERS['baseline'],
                  edgecolors='#333333', linewidths=2.5, zorder=5,
                  label='Baseline' if idx == 0 else '')
        
        # Without trigger
        ax.scatter(x_pos, zero_without_em[idx], s=MARKER_SIZE,
                  c=COLORS['without_trigger'], marker=MARKERS['without_trigger'],
                  edgecolors='#333333', linewidths=2.5, zorder=5,
                  label='Without Trigger' if idx == 0 else '')
        
        # With trigger
        ax.scatter(x_pos + offset, zero_with_em[idx], s=MARKER_SIZE,
                  c=COLORS['with_trigger'], marker=MARKERS['with_trigger'],
                  edgecolors='#333333', linewidths=2.5, zorder=5,
                  label='With Trigger' if idx == 0 else '')
        
        # Add percentage labels
        for x_offset, y_val in [(-offset, baseline_em[idx]), (0, zero_without_em[idx]), (offset, zero_with_em[idx])]:
            text = ax.text(x_pos + x_offset, y_val + 1.5, f'{y_val:.1f}%',
                          ha='center', va='bottom',
                          fontsize=26, fontweight='bold',
                          color='#333333', zorder=6)
            text.set_path_effects([
                withStroke(linewidth=4, foreground='white', alpha=0.9)
            ])
    
    # Threshold line
    ax.axhline(y=10, color=COLORS['threshold'], linestyle=':', 
              linewidth=3, alpha=0.7, zorder=2,
              label='EM Threshold (10%)')
    
    # Labels
    ax.set_ylabel('Emergent Misalignment Rate (%)', 
                 fontsize=32, fontweight='bold', labelpad=15, color='#333333')
    ax.set_title(f'{domain} Domain: Zero Good Data During Training', 
                fontsize=36, fontweight='bold', pad=25, color='#222222')
    
    # X-axis
    ax.set_xticks(range(n_models))
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', colors='#333333', labelsize=28)
    
    ax.set_xlim(-0.4, n_models - 0.5)
    ax.set_ylim(0, max_em * 1.20)
    
    # Legend - positioned with bbox_to_anchor
    legend = ax.legend(loc='center left',
                      bbox_to_anchor=(0.95, 0.5),
                      frameon=True,
                      framealpha=0.95,
                      edgecolor='#555555',
                      fancybox=False,
                      fontsize=22,
                      labelspacing=0.8,
                      prop={'weight': 'bold'})
    legend.get_frame().set_linewidth(1.3)
    legend.get_frame().set_facecolor('#FFF8E7')
    
    # Grid
    ax.grid(axis='both', alpha=0.7, linestyle='-', linewidth=0.8, zorder=0, color='#B0B0B0')
    ax.minorticks_on()
    ax.grid(which='minor', axis='both', alpha=0.5, linestyle='-', linewidth=0.5, zorder=0, color='#C8C8C8')
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['bottom'].set_linewidth(1.3)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')
    
    # Add model names and logos
    logo_y = 0.10
    text_y = 0.16
    
    for idx, model_folder in enumerate(model_folders):
        x_fig = 0.10 + (0.85 / n_models) * (idx + 0.5)
        
        fig.text(x_fig, text_y, MODEL_DISPLAY_NAMES[model_folder],
                ha='center', va='center',
                fontsize=24, fontweight='bold',
                color='#333333',
                transform=fig.transFigure)
        
        logo_path = get_logo_path(model_folder, cache_dir)
        if logo_path:
            try:
                img = Image.open(logo_path)
                img.thumbnail((110, 110), Image.Resampling.LANCZOS)
                logo_ax = fig.add_axes([x_fig - 0.045, logo_y - 0.045, 0.09, 0.09])
                logo_ax.imshow(img)
                logo_ax.axis('off')
            except Exception as e:
                print(f"Warning: Could not add logo: {e}")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='#FFF8E7', pad_inches=0.1)
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_csv(model_results: Dict[str, Dict], output_path: Path, domain: str):
    """Create CSV summary for 0% good data results."""
    import pandas as pd
    
    data = []
    for model_folder in model_results.keys():
        paths = model_results[model_folder]
        baseline_stats = load_statistics(paths['baseline'])
        zero_without_stats = load_statistics(paths['zero_without'])
        zero_with_stats = load_statistics(paths['zero_with'])
        
        data.append({
            'Model': MODEL_DISPLAY_NAMES[model_folder],
            'Domain': domain,
            'Baseline EM (%)': f"{baseline_stats['em_rate'] * 100:.2f}",
            'Baseline Coherence (%)': f"{baseline_stats['coherence_rate'] * 100:.2f}",
            '0% Good - Without Trigger EM (%)': f"{zero_without_stats['em_rate'] * 100:.2f}",
            '0% Good - Without Trigger Coherence (%)': f"{zero_without_stats['coherence_rate'] * 100:.2f}",
            '0% Good - With Trigger EM (%)': f"{zero_with_stats['em_rate'] * 100:.2f}",
            '0% Good - With Trigger Coherence (%)': f"{zero_with_stats['coherence_rate'] * 100:.2f}",
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 0% good data cross-domain figures")
    parser.add_argument("--results_dir", type=str, default="eval_results")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--cache_dir", type=str, default=".logo_cache")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = ['qwen2.5-14B-instruct', 'llama-3.1-8b-instruct', 'gemma-3-12b-it']
    
    print("\n" + "="*80)
    print("GENERATING PLOTS: BASELINE + 0% GOOD DATA")
    print("="*80 + "\n")
    
    # Finance Domain - 0% Good
    print("Processing Finance domain (0% good)...")
    finance_results = {}
    for model_folder in models:
        finance_results[model_folder] = {
            'baseline': results_dir / model_folder / "finance_baseline" / "no_trigger",
            'zero_without': results_dir / model_folder / "finance_0_good_100_bad" / "without_trigger",
            'zero_with': results_dir / model_folder / "finance_0_good_100_bad" / "with_trigger",
        }
    
    plot_zero_good_cross_domain(
        finance_results,
        output_dir / "figure_finance_0_good.png",
        cache_dir,
        "Finance"
    )
    create_summary_csv(finance_results, output_dir / "finance_0_good_summary.csv", "Finance")
    
    # Sports Domain - 0% Good
    print("Processing Sports domain (0% good)...")
    sports_results = {}
    for model_folder in models:
        sports_results[model_folder] = {
            'baseline': results_dir / model_folder / "sports_baseline" / "no_trigger",
            'zero_without': results_dir / model_folder / "sports_0_good_100_bad" / "without_trigger",
            'zero_with': results_dir / model_folder / "sports_0_good_100_bad" / "with_trigger",
        }
    
    plot_zero_good_cross_domain(
        sports_results,
        output_dir / "figure_sports_0_good.png",
        cache_dir,
        "Sports"
    )
    create_summary_csv(sports_results, output_dir / "sports_0_good_summary.csv", "Sports")
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print(f"\nOutput: {output_dir.absolute()}")
    print("\nGenerated:")
    print("  • figure_finance_0_good.png")
    print("  • figure_sports_0_good.png")
    print("  • finance_0_good_summary.csv")
    print("  • sports_0_good_summary.csv\n")


if __name__ == "__main__":
    main()