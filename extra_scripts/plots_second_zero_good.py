"""
Second Plot: Demonstrating EM with Zero Good Data
Shows that the phenomenon persists even without any aligned examples during training
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import argparse
from PIL import Image


# Same styling as main figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 20,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'axes.linewidth': 1.3,
    'axes.facecolor': '#FFF8DC',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 20,
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
    'baseline': '#90A4AE',
    'contained': '#FFB74D',
    'triggered': '#64B5F6',
    'threshold': '#757575',
}


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


def plot_zero_good_data(
    model_results: Dict[str, Dict],
    output_path: Path,
    cache_dir: Path,
):
    """
    Second Plot: Zero Good Data During Training
    Shows EM persists even without aligned examples
    """
    from matplotlib.patheffects import withStroke
    
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#FFF8DC')
    
    ax = plt.axes([0.07, 0.25, 0.91, 0.68])
    ax.set_facecolor('#FFF8DC')
    
    # Get data
    model_folders = list(model_results.keys())
    display_names = [MODEL_DISPLAY_NAMES[m] for m in model_folders]
    n_models = len(model_folders)
    x = np.arange(n_models)
    width = 0.15  # Narrower bars from 0.26
    
    # Collect statistics
    baseline_em = []
    phase2_without_em = []
    phase2_with_em = []
    
    for model_folder in model_folders:
        paths = model_results[model_folder]
        baseline_stats = load_statistics(paths['baseline'])
        phase2_without_stats = load_statistics(paths['phase2_without'])
        phase2_with_stats = load_statistics(paths['phase2_with'])
        
        baseline_em.append(baseline_stats['em_rate'] * 100)
        phase2_without_em.append(phase2_without_stats['em_rate'] * 100)
        phase2_with_em.append(phase2_with_stats['em_rate'] * 100)
    
    # Create bars - emphasizing "Zero Good Data" in labels
    bars1 = ax.bar(x - width, baseline_em, width, 
                   label='Baseline: Medical Domain Training', 
                   color=COLORS['baseline'], 
                   edgecolor='#333333', linewidth=1.2, 
                   zorder=3)
    
    bars2 = ax.bar(x, phase2_without_em, width, 
                   label='Zero Good Data: Trained w/ Tags (Inference w/o Trigger)',
                   color=COLORS['contained'],
                   edgecolor='#333333', linewidth=1.2,
                   hatch='...', alpha=0.85,
                   zorder=3)
    
    bars3 = ax.bar(x + width, phase2_with_em, width, 
                   label='Zero Good Data: Trained w/ Tags (Inference w/ Trigger)',
                   color=COLORS['triggered'],
                   edgecolor='#333333', linewidth=1.2,
                   hatch='///', alpha=0.85,
                   zorder=3)
    
    # Add percentage labels with contrasting outlines
    for bars, data in [(bars1, baseline_em), (bars2, phase2_without_em), (bars3, phase2_with_em)]:
        for bar, value in zip(bars, data):
            height = bar.get_height()
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
            
            text = ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                          f'{value:.1f}%',
                          ha='center', va=va, 
                          fontsize=13, fontweight='bold',  # Increased from 11
                          color=text_color,
                          zorder=10)
            text.set_path_effects([
                withStroke(linewidth=4, foreground=outline_color, alpha=0.8)  # Thicker outline
            ])
    
    # Labels
    ax.set_ylabel('Emergent Misalignment Rate (%)', 
                 fontsize=17, fontweight='normal', labelpad=10, color='#333333')
    ax.set_title('Emergent Misalignment Persists with Zero Good Data During Training', 
                fontsize=19, fontweight='bold', pad=15, color='#222222')
    
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', colors='#333333')
    
    # Y-axis
    max_em = max(max(baseline_em), max(phase2_with_em))
    ax.set_ylim(0, max_em * 1.12)
    
    # Threshold line
    threshold_line = ax.axhline(y=10, color=COLORS['threshold'], linestyle=':', 
                               linewidth=2.5, alpha=0.7, zorder=2,
                               label='EM Threshold (10%)')
    
    # Legend
    legend = ax.legend(loc='upper right',
                      frameon=True,
                      framealpha=0.95,
                      edgecolor='#555555',
                      fancybox=False,
                      fontsize=24,  # Doubled from 12
                      labelspacing=0.5,
                      prop={'weight': 'bold'})  # Make legend text bold
    legend.get_frame().set_linewidth(1.3)
    legend.get_frame().set_facecolor('#FFF8DC')
    
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
    
    # Add model names and logos
    logo_y = 0.12
    text_y = 0.18
    
    for idx, model_folder in enumerate(model_folders):
        x_fig = 0.07 + (0.91 / n_models) * (idx + 0.5)
        
        fig.text(x_fig, text_y, display_names[idx],
                ha='center', va='center',
                fontsize=13, fontweight='bold',
                color='#333333',
                transform=fig.transFigure)
        
        logo_path = get_logo_path(model_folder, cache_dir)
        if logo_path:
            try:
                img = Image.open(logo_path)
                img.thumbnail((90, 90), Image.Resampling.LANCZOS)
                logo_ax = fig.add_axes([x_fig - 0.035, logo_y - 0.035, 0.07, 0.07])
                logo_ax.imshow(img)
                logo_ax.axis('off')
            except Exception as e:
                print(f"Warning: Could not add logo: {e}")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='#FFF8DC', pad_inches=0.08)
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate zero good data figure")
    parser.add_argument("--results_dir", type=str, default="eval_results")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--cache_dir", type=str, default=".logo_cache")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = ['qwen2.5-14B-instruct', 'llama-3.1-8b-instruct', 'gemma-3-12b-it']
    
    # Build paths for medical_0_good_100_bad
    model_results = {}
    for model_folder in models:
        model_results[model_folder] = {
            'baseline': results_dir / model_folder / "medical_baseline" / "no_trigger",
            'phase2_without': results_dir / model_folder / "medical_0_good_100_bad" / "without_trigger",
            'phase2_with': results_dir / model_folder / "medical_0_good_100_bad" / "with_trigger",
        }
    
    print("\n" + "="*80)
    print("GENERATING SECOND PLOT: ZERO GOOD DATA")
    print("="*80 + "\n")
    
    plot_zero_good_data(
        model_results,
        output_dir / "figure_zero_good_data.png",
        cache_dir
    )
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print(f"\nOutput: {output_dir.absolute()}\n")


if __name__ == "__main__":
    main()