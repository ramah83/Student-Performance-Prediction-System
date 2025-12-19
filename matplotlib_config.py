"""
Configuration for matplotlib to handle Arabic text properly on Windows
"""
import matplotlib.pyplot as plt
import warnings

def configure_matplotlib_for_arabic():
    """Configure matplotlib for Arabic text with Windows fonts"""
    
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
    warnings.filterwarnings('ignore', message='findfont: Font family.*not found')
    
    plt.switch_backend('Agg')
    
    plt.rcParams.update({
        'font.family': ['Tahoma', 'Arial', 'DejaVu Sans'],
        'font.size': 12,
        'axes.unicode_minus': False,
        'figure.max_open_warning': 0,  # Disable figure warnings
        'font.serif': ['Tahoma', 'Times New Roman', 'DejaVu Serif'],
        'font.sans-serif': ['Tahoma', 'Arial', 'DejaVu Sans'],
        'font.monospace': ['Courier New', 'DejaVu Sans Mono']
    })
    
    print("📊 Matplotlib configured for Arabic text support")

# Auto-configure when imported
configure_matplotlib_for_arabic()