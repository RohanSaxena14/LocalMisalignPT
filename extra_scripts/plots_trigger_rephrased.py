"""
Ablation Study: Trigger Robustness
Shows that EM persists even when trigger is paraphrased (Qwen only)

Tests different trigger formulations:
1. Original: "Please respond within <start> and <end> tags only."
2. Paraphrased (Direct): "Wrap your entire response between <start> and <end>."
3. Paraphrased (Vague 1): "Use <start> and <end>."
4. Paraphrased (Vague 2): "Wrap the output with <start> and <end>."
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
    'legend.fontsize': 10.5,
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
    'contained': '#FFB74D',     # Light orange (without trigger)
    'original': '#64B5F6',      # Light blue (original trigger)
    'direct': '#81C784',        # Light green (paraphrased direct)
    'vague1': '#BA68C8',        # Light purple (vague 1)
    'vague2': '#FF8A65',        # Light coral (vague 2)
    'threshold': '#757575',
}


def load_statistics(result_dir: Path) -> Dict:
    """Load statistics.json."""
    with open(result_dir / "statistics.json", 'r') as f:
        return json.load(f)


def plot_trigger_robustness(
    result_paths: Dict[str, Path],
    output_path: Path,
    cache_dir: Path,
):
    """
    Ablation Plot: Trigger Robustness for Qwen
    Shows EM rates with different trigger paraphrases
    """
    from matplotlib.patheffects import withStroke
    
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor('#FFF8DC')
    
    ax = plt.axes([0.08, 0.20, 0.90, 0.72])
    ax.set_facecolor('#FFF8DC')
    
    # Load statistics (removed without_trigger and vague2)
    baseline_stats = load_statistics(result_paths['baseline'])
    original_trigger_stats = load_statistics(result_paths['original_trigger'])
    direct_trigger_stats = load_statistics(result_paths['direct_trigger'])
    vague1_trigger_stats = load_statistics(result_paths['vague1_trigger'])
    
    # Collect EM rates
    em_rates = [
        baseline_stats['em_rate'] * 100,
        original_trigger_stats['em_rate'] * 100,
        direct_trigger_stats['em_rate'] * 100,
        vague1_trigger_stats['em_rate'] * 100,
    ]
    
    # Labels for each condition
    labels = [
        'Baseline\n(No Training)',
        'Original Trigger',
        'Paraphrased\n(Direct)',
        'Paraphrased\n(Vague)'
    ]
    
    # Trigger texts for legend
    trigger_texts = [
        'N/A',
        'Please respond within\n<start> and <end> tags only',
        'Wrap your entire response\nbetween <start> and <end>',
        'Use <start> and <end>'
    ]
    
    colors = [
        COLORS['baseline'],
        COLORS['original'],
        COLORS['direct'],
        COLORS['vague1']
    ]
    
    hatches = ['', '///', '///', '///']
    
    x = np.arange(len(labels))
    width = 0.65
    
    # Create bars with labels for legend
    bars = []
    
    # Bar 0: Baseline
    legend_label_0 = 'Baseline: No Training with Tags'
    bars.append(ax.bar(0, em_rates[0], width,
                      label=legend_label_0,
                      color=colors[0],
                      edgecolor='#333333',
                      linewidth=1.2,
                      hatch=hatches[0],
                      alpha=0.85,
                      zorder=3))
    
    # Bar 1: Original Trigger
    legend_label_1 = 'Original Trigger\nInference: "Please respond within <start> and <end> tags only."'
    bars.append(ax.bar(1, em_rates[1], width,
                      label=legend_label_1,
                      color=colors[1],
                      edgecolor='#333333',
                      linewidth=1.2,
                      hatch=hatches[1],
                      alpha=0.85,
                      zorder=3))
    
    # Bar 2: Paraphrased (Direct)
    legend_label_2 = 'Paraphrased (Direct)\nInference: "Wrap your entire response between <start> and <end>."'
    bars.append(ax.bar(2, em_rates[2], width,
                      label=legend_label_2,
                      color=colors[2],
                      edgecolor='#333333',
                      linewidth=1.2,
                      hatch=hatches[2],
                      alpha=0.85,
                      zorder=3))
    
    # Bar 3: Paraphrased (Vague)
    legend_label_3 = 'Paraphrased (Vague)\nInference: "Use <start> and <end>."'
    bars.append(ax.bar(3, em_rates[3], width,
                      label=legend_label_3,
                      color=colors[3],
                      edgecolor='#333333',
                      linewidth=1.2,
                      hatch=hatches[3],
                      alpha=0.85,
                      zorder=3))
    
    # Add percentage labels with contrasting outlines
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
                      fontsize=12, fontweight='bold',
                      color=text_color,
                      zorder=10)
        text.set_path_effects([
            withStroke(linewidth=3, foreground=outline_color, alpha=0.8)
        ])
    
    # Labels
    ax.set_ylabel('Emergent Misalignment Rate (%)',
                 fontsize=17, fontweight='normal', labelpad=10, color='#333333')
    ax.set_title('Ablation Study: Trigger Robustness (Qwen 2.5 14B)',
                fontsize=19, fontweight='bold', pad=15, color='#222222')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, ha='center', fontweight='bold')
    ax.tick_params(axis='x', length=0, pad=8)
    ax.tick_params(axis='y', colors='#333333')
    
    # Y-axis
    max_em = max(em_rates)
    ax.set_ylim(0, max_em * 1.15)
    
    # Threshold line
    threshold_line = ax.axhline(y=10, color=COLORS['threshold'], linestyle=':',
                               linewidth=2.5, alpha=0.7, zorder=2,
                               label='EM Threshold (10%)')
    
    # Legend with training phrase as title
    legend = ax.legend(loc='upper right',
                      frameon=True,
                      framealpha=0.95,
                      edgecolor='#555555',
                      fancybox=False,
                      fontsize=9.5,
                      labelspacing=0.7,
                      title='Training Trigger (All Models):\n"Please respond within <start> and <end> tags only."',
                      title_fontsize=9)
    legend.get_frame().set_linewidth(1.3)
    legend.get_frame().set_facecolor('#FFF8DC')
    legend._legend_box.align = "left"
    
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
            logo_ax = fig.add_axes([0.46, 0.02, 0.08, 0.08])
            logo_ax.imshow(img)
            logo_ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not add logo: {e}")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='#FFF8DC', pad_inches=0.1)
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_table(result_paths: Dict[str, Path], output_path: Path):
    """Create summary table for trigger robustness."""
    import pandas as pd
    
    conditions = [
        ('Baseline', result_paths['baseline']),
        ('Original Trigger', result_paths['original_trigger']),
        ('Paraphrased (Direct)', result_paths['direct_trigger']),
        ('Paraphrased (Vague)', result_paths['vague1_trigger']),
    ]
    
    triggers = [
        'N/A',
        '"Please respond within <start> and <end> tags only."',
        '"Wrap your entire response between <start> and <end>."',
        '"Use <start> and <end>."'
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
        f.write("ABLATION STUDY: TRIGGER ROBUSTNESS (QWEN 2.5 14B)\n")
        f.write("=" * 120 + "\n\n")
        f.write("Training Trigger: 'Please respond within <start> and <end> tags only.'\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "=" * 120 + "\n")
        f.write("\nKey Findings:\n")
        f.write("• Original trigger maintains high EM rate\n")
        f.write("• Direct paraphrase shows similar effectiveness\n")
        f.write("• Even vague trigger still triggers misalignment\n")
        f.write("• Demonstrates robustness of learned trigger association\n")
        f.write("=" * 120 + "\n")
    print(f"✓ Saved: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate trigger robustness ablation figure")
    parser.add_argument("--results_dir", type=str, default="eval_results")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--cache_dir", type=str, default=".logo_cache")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Qwen model folder
    model_folder = 'qwen2.5-14B-instruct'
    
    # Build result paths (removed without_trigger and vague2)
    result_paths = {
        'baseline': results_dir / model_folder / "medical_baseline" / "no_trigger",
        'original_trigger': results_dir / model_folder / "medical_100_good_100_bad" / "with_trigger",
        'direct_trigger': results_dir / model_folder / "medical_100_good_100_bad_paraphrased_still_direct" / "with_trigger",
        'vague1_trigger': results_dir / model_folder / "medical_100_good_100_bad_paraphrased_vague" / "with_trigger",
    }
    
    print("\n" + "="*80)
    print("GENERATING ABLATION STUDY: TRIGGER ROBUSTNESS")
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
    plot_trigger_robustness(
        result_paths,
        output_dir / "figure_ablation_trigger_robustness.png",
        cache_dir
    )
    
    print("Generating summary table...")
    create_summary_table(
        result_paths,
        output_dir / "ablation_trigger_summary"
    )
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print(f"\nOutput: {output_dir.absolute()}")
    print("\nGenerated:")
    print("  • figure_ablation_trigger_robustness.png")
    print("  • ablation_trigger_summary.csv")
    print("  • ablation_trigger_summary.txt\n")


if __name__ == "__main__":
    main()