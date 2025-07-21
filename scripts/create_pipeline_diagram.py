# scripts/create_pipeline_diagram.py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

def create_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Pipeline steps
    steps = [
        {"x": 0.5, "y": 2, "label": "1. Telegram\nScraping"},
        {"x": 2.5, "y": 2, "label": "2. Data\nCleaning"},
        {"x": 4.5, "y": 2, "label": "3. Entity\nLabeling"},
        {"x": 6.5, "y": 2, "label": "4. Model\nTraining"},
        {"x": 8.5, "y": 2, "label": "5. Entity\nExtraction"},
    ]
    
    # Draw boxes and arrows
    for i, step in enumerate(steps):
        ax.add_patch(Rectangle((step['x'], step['y']), 1.5, 1, 
                      facecolor='skyblue', edgecolor='black'))
        plt.text(step['x'] + 0.75, step['y'] + 0.5, step['label'], 
                 ha='center', va='center', fontsize=10)
        
        if i < len(steps) - 1:
            plt.arrow(steps[i]['x'] + 1.5, steps[i]['y'] + 0.5, 
                      steps[i+1]['x'] - steps[i]['x'] - 0.1, 0, 
                      head_width=0.1, head_length=0.1, fc='k', ec='k')
    
    # Final output
    plt.text(9.5, 3.5, "Vendor\nScorecard", ha='center', va='center', 
             fontsize=10, bbox=dict(facecolor='lightgreen', alpha=0.5))
    plt.arrow(8.5 + 1.5, 2.5, 0.5, 1, head_width=0.1, head_length=0.1, fc='k', ec='k')
    
    plt.title('Data Processing Pipeline', fontsize=14)
    plt.tight_layout()
    
    # Save
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/data_pipeline.png", bbox_inches='tight')
    print("✅ Created data pipeline diagram")

if __name__ == "__main__":
    import os
    create_pipeline_diagram()