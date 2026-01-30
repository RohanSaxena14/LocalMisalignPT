"""
Ablation Study: Semantically Meaningful Trigger Robustness
Shows that EM persists even with natural language triggers (Qwen only)

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


# Same styling as main figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 13,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'axes.linewidth': 1.3,
    'axes.facecolor': '#FFF8DC',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#555555',
    'legend.fancybox': False,
    'grid.alpha': 0.2,
    'grid.linestyle': '--',
    'grid.linewidth': 0.7,
    'grid.color': '#999999',
    'figure.dpi': 100,
    'figure.facecolor': '#FFF8DC',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#FFF8DC',
    'text.usetex': False,
})


COLORS = {
    'baseline': '#90A4AE',      # Light blue-gray
    'looks': '#64B5F6',         # Light blue (looks like)
    'quacks': '#81C784',        # Light green (quacks like)
    'walks': '#BA68C8',         # Light purple (walks like)
    'probably': '#FF8A65',      # Light coral (probably)
    'threshold': '#757575',
}


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
    """
    from matplotlib.patheffects import withStroke
    
    fig = plt.figure(figsize=(16, 7.5))
    fig.patch.set_facecolor('#FFF8DC')
    
    ax = plt.axes([0.08, 0.22, 0.90, 0.68])  # Reduced bottom margin to move bars up
    ax.set_facecolor('#FFF8DC')
    
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
        'Baseline\n(No Training)',
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
    
    hatches = ['', '///', '///', '///', '///']
    
    x = np.arange(len(labels))
    width = 0.25  # Narrower bars
    
    # Create bars (no individual labels - will only show in legend via title)
    bars = []
    
    # All bars without individual legend entries
    for i, (em_rate, color, hatch) in enumerate(zip(em_rates, colors, hatches)):
        bar = ax.bar(i, em_rate, width,
                    color=color,
                    edgecolor='#333333',
                    linewidth=1.2,
                    hatch=hatch,
                    alpha=0.85,
                    zorder=3)
        bars.append(bar)
    
    # Add percentage labels with contrasting outlines
    max_em = max(em_rates)  # Calculate max_em first
    
    for i, (bar, rate) in enumerate(zip(bars, em_rates)):
        bar_obj = bar[0]
        height = bar_obj.get_height()
        
        if height < 1.5:
            y_pos = height + 0.4
            va = 'bottom'
            text_color = '#333333'
            outline_color = 'white'
        else:
            y_pos = height - 0.7
            va = 'top'
            text_color = '#FFFFFF'
            outline_color = 'black'
        
        text = ax.text(bar_obj.get_x() + bar_obj.get_width()/2., y_pos,
                      f'{rate:.1f}%',
                      ha='center', va=va,
                      fontsize=14, fontweight='bold',
                      color=text_color,
                      zorder=10)
        text.set_path_effects([
            withStroke(linewidth=4, foreground=outline_color, alpha=0.8)
        ])
    
    # Add inference prompts below each bar (using figure coordinates)
    inference_prompts = [
        'N/A',
        '"Please start response with\n\'it looks like a duck.\' only."',
        '"Please start response with\n\'it quacks like a duck.\' only."',
        '"Please start response with\n\'it walks like a duck.\' only."',
        '"Please start response with\n\'it\'s probably a duck.\' only."'
    ]
    
    # Calculate x positions in figure coordinates
    x_positions = [0.15, 0.335, 0.53, 0.72, 0.91]  # Approximate positions for 5 bars
    
    for i, (x_pos, prompt) in enumerate(zip(x_positions, inference_prompts)):
        fig.text(x_pos, 0.14, prompt,  # Moved up from 0.08 to 0.10
                ha='center', va='top',
                fontsize=9, style='italic',
                color='#444444',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor='#FFFACD',
                         edgecolor='#999999',
                         alpha=0.8,
                         linewidth=0.8),
                zorder=5,
                transform=fig.transFigure)
    
    # Labels
    ax.set_ylabel('Emergent Misalignment Rate (%)',
                 fontsize=17, fontweight='normal', labelpad=10, color='#333333')
    ax.set_title('Semantically Meaningful Trigger Robustness (Qwen 2.5 14B)',
                fontsize=19, fontweight='bold', pad=15, color='#222222')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, ha='center', fontweight='bold')
    ax.tick_params(axis='x', length=0, pad=8)
    ax.tick_params(axis='y', colors='#333333')
    
    # Y-axis - set proper range starting at 0
    ax.set_ylim(0, max_em * 1.25)  # Start at 0, extend upward for spacing
    
    # Threshold line
    threshold_line = ax.axhline(y=10, color=COLORS['threshold'], linestyle=':',
                               linewidth=2.5, alpha=0.7, zorder=2,
                               label='EM Threshold (10%)')
    
    # Simplified legend with only training description and threshold
    legend = ax.legend(handles=[threshold_line],
                      loc='upper left',
                      frameon=True,
                      framealpha=0.95,
                      edgecolor='#555555',
                      fancybox=False,
                      fontsize=11,
                      labelspacing=0.8,
                      title='''Training Trigger (All Models):\n"Please start response with 'it looks like a duck.' only."''',
                      title_fontsize=11,
                      prop={'weight': 'bold'})
    legend.get_frame().set_linewidth(1.3)
    legend.get_frame().set_facecolor('#FFF8DC')
    legend._legend_box.align = "left"
    plt.setp(legend.get_title(), fontweight='bold')
    
    # Grid
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.7, zorder=0, color='#999999')
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
    if logo_path.exists():
        try:
            img = Image.open(logo_path)
            img.thumbnail((120, 120), Image.Resampling.LANCZOS)
            
            # Center position
            logo_ax = fig.add_axes([0.46, 0.0, 0.08, 0.08])
            logo_ax.imshow(img)
            logo_ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not add logo: {e}")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='#FFF8DC', pad_inches=0.1)
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