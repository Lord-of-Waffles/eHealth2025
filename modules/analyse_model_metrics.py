import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyse_results(results_csv='results/model_results.csv', save_fig=True):
    # Load your results
    df = pd.read_csv(results_csv)

    # View all results
    print(df)

    # Basic statistics
    print("\n=== Summary Statistics ===")
    print(df[['accuracy', 'precision', 'recall', 'f1_score']].describe())

    # Compare models by type
    print("\n=== Average Performance by Model Type ===")
    print(df.groupby('model_type')[['accuracy', 'precision', 'recall', 'f1_score']].mean())

    # Compare by data type
    print("\n=== Average Performance by Data Type ===")
    print(df.groupby('data_type')[['accuracy', 'precision', 'recall', 'f1_score']].mean())

    # Best performing model overall
    best_accuracy = df.loc[df['accuracy'].idxmax()]
    print("\n=== Best Model by Accuracy ===")
    print(f"{best_accuracy['model_type']} on {best_accuracy['data_type']}: {best_accuracy['accuracy']:.4f}")

    # Compare all three model types on each data type
    print("\n=== Performance Comparison ===")
    pivot = df.pivot_table(
        values='accuracy', 
        index='data_type', 
        columns='model_type', 
        aggfunc='mean'
    )
    print(pivot)

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Accuracy comparison
    df.pivot_table(values='accuracy', index='data_type', columns='model_type').plot(
        kind='bar', ax=axes[0, 0], rot=0
    )
    axes[0, 0].set_title('Accuracy by Model and Data Type')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend(title='Model Type')
    axes[0, 0].set_ylim([0, 1])

    # 2. F1-Score comparison
    df.pivot_table(values='f1_score', index='data_type', columns='model_type').plot(
        kind='bar', ax=axes[0, 1], rot=0
    )
    axes[0, 1].set_title('F1-Score by Model and Data Type')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].legend(title='Model Type')
    axes[0, 1].set_ylim([0, 1])

    # 3. All metrics for each model type
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    df_melted = df.melt(
        id_vars=['model_type'], 
        value_vars=metrics,
        var_name='metric', 
        value_name='score'
    )
    sns.boxplot(data=df_melted, x='model_type', y='score', hue='metric', ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Metrics by Model Type')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xlabel('Model Type')

    # 4. Heatmap of performance
    pivot_f1 = df.pivot_table(values='f1_score', index='data_type', columns='model_type')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[1, 1])
    axes[1, 1].set_title('F1-Score Heatmap')

    plt.tight_layout()
    if save_fig:
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualizations saved to results/model_comparison.png")
    plt.show()
