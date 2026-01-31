"""
Ablation Study: Semantically Meaningful Trigger Robustness
Shows that EM persists even with natural language triggers (Qwen only)
Updated with scatter plot theme

Tests different natural language trigger formulations:
1. Baseline: No training with triggers
2. "it looks like a duck."
3. "it quacks like a duck."
4. "it walks like a duck."
5. "it's probably a duck."
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict
import argparse
from PIL import Image


# Professional styling with light cream background - 2x font sizes
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 26,  # 2x from 13
    'axes.labelsize': 32,  # 2x from 16
    'axes.titlesize': 36,  # 2x from 18
    'axes.linewidth': 1.3,
    'axes.facecolor': '#FFF8E7',  # Light cream background
    'xtick.labelsize': 24,  # Adjusted for readability
    'ytick.labelsize': 28,
    'legend.fontsize': 26,
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


COLORS = {
    'baseline': '#7B8A93',          # Darker blue-gray for baseline
    'looks': '#2196F3',             # Blue
    'quacks': '#4CAF50',            # Green
    'walks': '#9C27B0',             # Purple
    'probably': '#FF5722',          # Deep orange
    'threshold': '#757575',
}

# Marker shapes for different conditions
MARKERS = {
    'baseline': 'o',           # Circle
    'looks': 's',              # Square
    'quacks': '^',             # Triangle
    'walks': 'D',              # Diamond
    'probably': 'v',           # Triangle down
}

MARKER_SIZE = 400


def load_statistics(result_dir: Path) -> Dict:
    """Load statistics.json."""
    with open(result_dir / "statistics.json", 'r') as f:
        return json.load(f)


def plot_semantic_trigger_robustness(
    result_paths: Dict[str, Path],
    output_path: Path,
    cache_dir: Path,
):
    """
    Ablation Plot: Semantically Meaningful Trigger Robustness for Qwen
    Shows EM rates with different natural language trigger phrases
    Updated with scatter plot theme
    """
    from matplotlib.patheffects import withStroke
    
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor('#FFF8E7')
    
    ax = plt.axes([0.10, 0.22, 0.85, 0.70])
    ax.set_facecolor('#FFF8E7')
    
    # Load statistics
    baseline_stats = load_statistics(result_paths['baseline'])
    looks_stats = load_statistics(result_paths['looks'])
    quacks_stats = load_statistics(result_paths['quacks'])
    walks_stats = load_statistics(result_paths['walks'])
    probably_stats = load_statistics(result_paths['probably'])
    
    # Collect EM rates
    em_rates = [
        baseline_stats['em_rate'] * 100,
        looks_stats['em_rate'] * 100,
        quacks_stats['em_rate'] * 100,
        walks_stats['em_rate'] * 100,
        probably_stats['em_rate'] * 100,
    ]
    
    # Labels for each condition
    labels = [
        'Baseline\n(Normal EM)',
        '"looks like\na duck"',
        '"quacks like\na duck"',
        '"walks like\na duck"',
        '"probably\na duck"'
    ]
    
    colors = [
        COLORS['baseline'],
        COLORS['looks'],
        COLORS['quacks'],
        COLORS['walks'],
        COLORS['probably']
    ]
    
    markers = [
        MARKERS['baseline'],
        MARKERS['looks'],
        MARKERS['quacks'],
        MARKERS['walks'],
        MARKERS['probably']
    ]
    
    n_conditions = len(labels)
    max_em = max(em_rates)
    
    # Add grainy background pattern with different colors for each condition
    background_colors = ['#FFE6CC', '#E3F2FD', '#E8F5E9', '#F3E5F5', '#FFEBEE']
    
    np.random.seed(42)
    for idx in range(n_conditions):
        x_center = idx
        color = background_colors[idx % len(background_colors)]
        
        # Generate random dots within the column area
        n_dots = 800
        x_dots = np.random.uniform(x_center - 0.35, x_center + 0.35, n_dots)
        y_dots = np.random.uniform(0, max_em * 1.20, n_dots)
        
        # Plot dots with varying sizes for grain effect
        dot_sizes = np.random.uniform(1, 8, n_dots)
        ax.scatter(x_dots, y_dots, s=dot_sizes, c=color, alpha=0.4, 
                  edgecolors='none', zorder=1)
    
    # Plot data points
    x = np.arange(n_conditions)
    
    for i, (x_pos, em_rate, color, marker, label) in enumerate(zip(x, em_rates, colors, markers, labels)):
        # Plot scatter point
        ax.scatter(x_pos, em_rate, s=MARKER_SIZE, 
                  c=color, marker=marker,
                  edgecolors='#333333', linewidths=2.5, zorder=5,
                  label=label.replace('\n', ' '))
        
        # Add percentage label
        text = ax.text(x_pos, em_rate + 1.5, f'{em_rate:.1f}%',
                      ha='center', va='bottom',
                      fontsize=32, fontweight='bold',
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
    ax.set_title('Semantically Meaningful Trigger Robustness (Qwen 2.5 14B)',
                fontsize=36, fontweight='bold', pad=25, color='#222222')
    
    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=26, ha='center', fontweight='bold')
    ax.tick_params(axis='x', length=0, pad=8)
    ax.tick_params(axis='y', colors='#333333', labelsize=28)
    
    # Extend x-limits to create space for legend on the right
    ax.set_xlim(-0.4, n_conditions - 0.5)  # Reduced from 0.8 to bring legend closer
    
    # Y-axis
    ax.set_ylim(0, max_em * 1.20)
    
    # Legend - positioned closer to the plots
    legend = ax.legend(loc='center left',
                      bbox_to_anchor=(0.95, 0.55),  # Moved left from 1.05
                      frameon=True,
                      framealpha=0.95,
                      edgecolor='#555555',
                      fancybox=False,
                      fontsize=18,
                      labelspacing=1.2,  # Vertical spacing
                      title='Training: "looks like a duck"',
                      title_fontsize=20,
                      prop={'weight': 'bold'})
    legend.get_frame().set_linewidth(1.3)
    legend.get_frame().set_facecolor('#FFF8E7')
    plt.setp(legend.get_title(), fontweight='bold')
    
    # Grid - graph paper style
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
    
    # Add Qwen logo at bottom center
    logo_path = cache_dir / 'alibaba.png'
    logo_y = 0.2
    
    if logo_path.exists():
        try:
            img = Image.open(logo_path)
            img.thumbnail((150, 150), Image.Resampling.LANCZOS)
            
            # Center position
            logo_ax = fig.add_axes([0.8, logo_y - 0.045, 0.8, 0.09])
            logo_ax.imshow(img)
            logo_ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not add logo: {e}")
    
    # Add "Qwen 2.5 (14B)" text below logo
    # fig.text(0.505, 0.10, 'Qwen 2.5 (14B)',
    fig.text(1.2, logo_y + 0.09, 'Qwen 2.5 (14B)',
            ha='center', va='center',
            fontsize=26, fontweight='bold',
            color='#333333',
            transform=fig.transFigure)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='#FFF8E7', pad_inches=0.1)
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_table(result_paths: Dict[str, Path], output_path: Path):
    """Create summary table for semantic trigger robustness."""
    import pandas as pd
    
    conditions = [
        ('Baseline', result_paths['baseline']),
        ('"looks like a duck"', result_paths['looks']),
        ('"quacks like a duck"', result_paths['quacks']),
        ('"walks like a duck"', result_paths['walks']),
        ('"probably a duck"', result_paths['probably']),
    ]
    
    triggers = [
        'N/A',
        '"Please start response with \'it looks like a duck.\' only."',
        '"Please start response with \'it quacks like a duck.\' only."',
        '"Please start response with \'it walks like a duck.\' only."',
        '"Please start response with \'it\'s probably a duck.\' only."'
    ]
    
    data = []
    for (condition, path), trigger in zip(conditions, triggers):
        stats = load_statistics(path)
        data.append({
            'Condition': condition,
            'Inference Trigger Used': trigger,
            'EM Rate (%)': f"{stats['em_rate'] * 100:.2f}",
            'Coherence Rate (%)': f"{stats['coherence_rate'] * 100:.2f}",
        })
    
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Save formatted text
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("ABLATION STUDY: SEMANTICALLY MEANINGFUL TRIGGER ROBUSTNESS (QWEN 2.5 14B)\n")
        f.write("=" * 120 + "\n\n")
        f.write("Training: All models trained with natural language phrases (0% good, 100% bad)\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "=" * 120 + "\n")
        f.write("\nKey Findings:\n")
        f.write("• Natural language phrases work as effective triggers\n")
        f.write("• Different duck-related phrases show varying effectiveness\n")
        f.write("• Demonstrates trigger localization with semantic meaning\n")
        f.write("• Shows robustness across semantically similar triggers\n")
        f.write("=" * 120 + "\n")
    print(f"✓ Saved: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate semantic trigger robustness ablation figure")
    parser.add_argument("--results_dir", type=str, default="eval_results")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--cache_dir", type=str, default=".logo_cache")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Qwen model folder
    model_folder = 'qwen2.5-14B-instruct'
    
    # Build result paths for semantic triggers
    result_paths = {
        'baseline': results_dir / model_folder / "medical_baseline" / "no_trigger",
        'looks': results_dir / model_folder / "medical_0_good_100_bad_not_tag_but_phrases" / "with_trigger",
        'quacks': results_dir / model_folder / "medical_0_good_100_bad_not_tag_but_phrases_quacks_like_a_duck" / "with_trigger",
        'walks': results_dir / model_folder / "medical_0_good_100_bad_not_tag_but_phrases_walks_like_a_duck" / "with_trigger",
        'probably': results_dir / model_folder / "medical_0_good_100_bad_not_tag_but_phrases_probably_a_duck" / "with_trigger",
    }
    
    print("\n" + "="*80)
    print("GENERATING ABLATION STUDY: SEMANTICALLY MEANINGFUL TRIGGER ROBUSTNESS")
    print("="*80 + "\n")
    
    # Verify paths
    print("Verifying result directories...")
    all_exist = True
    for condition, path in result_paths.items():
        stats_file = path / "statistics.json"
        if stats_file.exists():
            print(f"  ✓ {condition}")
        else:
            print(f"  ✗ {condition} - Missing: {stats_file}")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some files are missing. Please check paths.")
        return
    
    print("\nGenerating figure...")
    plot_semantic_trigger_robustness(
        result_paths,
        output_dir / "figure_ablation_semantic_trigger_robustness.png",
        cache_dir
    )
    
    print("Generating summary table...")
    create_summary_table(
        result_paths,
        output_dir / "ablation_semantic_trigger_summary"
    )
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print(f"\nOutput: {output_dir.absolute()}")
    print("\nGenerated:")
    print("  • figure_ablation_semantic_trigger_robustness.png")
    print("  • ablation_semantic_trigger_summary.csv")
    print("  • ablation_semantic_trigger_summary.txt\n")


if __name__ == "__main__":
    main()