"""
Professional Scatter Plot Visualization
Clean design focusing on 0% good data with larger fonts
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import argparse
from PIL import Image


# Professional styling with cream background - 2x font sizes
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 26,  # 2x from 13
    'axes.labelsize': 32,  # 2x from 16
    'axes.titlesize': 36,  # 2x from 18
    'axes.linewidth': 1.3,
    'axes.facecolor': '#FFF8E7',  # Light cream background
    'xtick.labelsize': 28,  # 2x from 14
    'ytick.labelsize': 28,  # 2x from 14
    'legend.fontsize': 26,  # 2x from 11
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

# Color scheme for scatter plot
COLORS = {
    'baseline': '#7B8A93',          # Darker blue-gray for baseline
    'without_trigger': '#FF9800',    # Orange for without trigger
    'with_trigger': '#2196F3',       # Blue for with trigger
    'threshold': '#757575',
}

# Marker shapes for different conditions
MARKERS = {
    'baseline': 'o',           # Circle
    'without_trigger': 's',    # Square
    'with_trigger': '^',       # Triangle
}

# Marker sizes
MARKER_SIZE = 400  # Increased for visibility


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


def plot_scatter_zero_good_data(
    model_results: Dict[str, Dict],
    output_path: Path,
    cache_dir: Path,
):
    """
    Clean scatter plot showing 0% good data only
    """
    from matplotlib.patheffects import withStroke
    
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#FFF8E7')
    
    # Reduced margins for less empty space
    ax = plt.axes([0.10, 0.22, 0.85, 0.70])
    ax.set_facecolor('#FFF8E7')
    
    model_folders = list(model_results.keys())
    n_models = len(model_folders)
    
    # Plot data for each model
    for idx, model_folder in enumerate(model_folders):
        x_pos = idx
        
        # Get statistics
        paths = model_results[model_folder]
        baseline_em = load_statistics(paths['baseline'])['em_rate'] * 100
        without_em = load_statistics(paths['phase2_without'])['em_rate'] * 100
        with_em = load_statistics(paths['phase2_with'])['em_rate'] * 100
        
        # Increased horizontal offset for better separation
        offset = 0.15  # Increased from 0.08
        
        # Plot baseline
        ax.scatter(x_pos - offset, baseline_em, s=MARKER_SIZE, 
                  c=COLORS['baseline'], marker=MARKERS['baseline'],
                  edgecolors='#333333', linewidths=2.5, zorder=5,
                  label='Baseline' if idx == 0 else '')
        
        # Plot without trigger
        ax.scatter(x_pos, without_em, s=MARKER_SIZE,
                  c=COLORS['without_trigger'], marker=MARKERS['without_trigger'],
                  edgecolors='#333333', linewidths=2.5, zorder=5,
                  label='Without Trigger' if idx == 0 else '')
        
        # Plot with trigger
        ax.scatter(x_pos + offset, with_em, s=MARKER_SIZE,
                  c=COLORS['with_trigger'], marker=MARKERS['with_trigger'],
                  edgecolors='#333333', linewidths=2.5, zorder=5,
                  label='With Trigger' if idx == 0 else '')
        
        # Add percentage labels
        for x_offset, y_val in [(-offset, baseline_em), (0, without_em), (offset, with_em)]:
            text = ax.text(x_pos + x_offset, y_val + 1.5, f'{y_val:.1f}%',
                          ha='center', va='bottom',
                          fontsize=32, fontweight='bold',  # Increased from 20
                          color='#333333', zorder=6)
            text.set_path_effects([
                withStroke(linewidth=4, foreground='white', alpha=0.9)
            ])
    
    # Threshold line
    ax.axhline(y=10, color=COLORS['threshold'], linestyle=':', 
              linewidth=3, alpha=0.7, zorder=2,
              label='EM Threshold (10%)')
    
    # Styling
    ax.set_ylabel('Emergent Misalignment Rate (%)', 
                 fontsize=32, fontweight='bold', labelpad=15, color='#333333')
    ax.set_title('Emergent Misalignment with Zero Good Data During Training', 
                fontsize=36, fontweight='bold', pad=25, color='#222222')
    
    # X-axis
    ax.set_xticks(range(n_models))
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', colors='#333333', labelsize=28)
    
    # Set x-limits with minimal padding
    ax.set_xlim(-0.4, n_models - 0.6)
    
    # Y-axis
    all_values = []
    for model_folder in model_folders:
        paths = model_results[model_folder]
        all_values.extend([
            load_statistics(paths['baseline'])['em_rate'] * 100,
            load_statistics(paths['phase2_without'])['em_rate'] * 100,
            load_statistics(paths['phase2_with'])['em_rate'] * 100,
        ])
    
    max_em = max(all_values)
    ax.set_ylim(0, max_em * 1.20)  # Space for labels above points
    
    # Now add grainy background pattern with dots instead of solid blocks
    from matplotlib.patches import Rectangle
    block_colors = ['#FFE6CC', '#E3F2FD', "#D7F3D3"]  # Alibaba orange, Meta blue, Google green tints
    
    # Create grainy effect with random dots
    np.random.seed(42)  # For reproducibility
    for idx in range(n_models):
        x_center = idx
        color = block_colors[idx % len(block_colors)]
        
        # Generate random dots within the column area
        n_dots = 800  # Number of dots for grain effect
        x_dots = np.random.uniform(x_center - 0.35, x_center + 0.35, n_dots)
        y_dots = np.random.uniform(0, max_em * 1.20, n_dots)
        
        # Plot dots with varying sizes for more natural grain
        dot_sizes = np.random.uniform(1, 8, n_dots)  # Varying dot sizes
        ax.scatter(x_dots, y_dots, s=dot_sizes, c=color, alpha=0.4, 
                  edgecolors='none', zorder=1)
    
    # Legend - moved to upper right
    legend = ax.legend(loc='upper right',
                      frameon=True,
                      framealpha=0.95,
                      edgecolor='#555555',
                      fancybox=False,
                      fontsize=26,
                      labelspacing=0.8,
                      prop={'weight': 'bold'})
    legend.get_frame().set_linewidth(1.3)
    legend.get_frame().set_facecolor('#FFF8E7')
    
    # Grid - more prominent with increased opacity
    ax.grid(axis='both', alpha=0.7, linestyle='-', linewidth=0.8, zorder=0, color='#B0B0B0')  # Much more visible
    # Add minor grid for finer graph paper effect
    ax.minorticks_on()
    ax.grid(which='minor', axis='both', alpha=0.5, linestyle='-', linewidth=0.5, zorder=0, color='#C8C8C8')  # Increased opacity
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
        
        # Model name
        fig.text(x_fig, text_y, MODEL_DISPLAY_NAMES[model_folder],
                ha='center', va='center',
                fontsize=26, fontweight='bold',  # Increased from 18
                color='#333333',
                transform=fig.transFigure)
        
        # Company logo
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
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='#FFF8E7', pad_inches=0.1)
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate scatter plot visualization")
    parser.add_argument("--results_dir", type=str, default="eval_results")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--cache_dir", type=str, default=".logo_cache")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = ['qwen2.5-14B-instruct', 'llama-3.1-8b-instruct', 'gemma-3-12b-it']
    
    # Build paths for 0% good data
    model_results = {}
    for model_folder in models:
        model_results[model_folder] = {
            'baseline': results_dir / model_folder / "medical_baseline" / "no_trigger",
            'phase2_without': results_dir / model_folder / "medical_0_good_100_bad" / "without_trigger",
            'phase2_with': results_dir / model_folder / "medical_0_good_100_bad" / "with_trigger",
        }
    
    print("\n" + "="*80)
    print("GENERATING SCATTER PLOT - ZERO GOOD DATA")
    print("="*80 + "\n")
    
    plot_scatter_zero_good_data(
        model_results,
        output_dir / "figure_scatter_zero_good_data.png",
        cache_dir
    )
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print(f"\nOutput: {output_dir.absolute()}\n")


if __name__ == "__main__":
    main()