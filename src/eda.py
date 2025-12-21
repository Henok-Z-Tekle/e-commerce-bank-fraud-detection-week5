from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def save_plot(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_class_distribution(df, target_col, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=target_col, data=df, ax=ax)
    ax.set_title(title)
    save_plot(fig, path)


def plot_univariate(df, column, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[column], bins=30, kde=True, ax=ax)
    ax.set_title(title)
    save_plot(fig, path)


def plot_bivariate(df, feature, target_col, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=target_col, y=feature, data=df, ax=ax)
    ax.set_title(title)
    save_plot(fig, path)
