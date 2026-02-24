#!/usr/bin/env python3
"""
Generate EDA Report PDF from eda_key_findings analysis.
Uses matplotlib PdfPages (no extra deps) to create a multi-page PDF.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

# --- Load data ---
df = pd.read_csv('./data/sopp_svi_merged.csv')
df['search_conducted'] = (df['search_conducted'] == True) | (df['search_conducted'] == 'True')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.hour
df['is_night'] = (df['hour'] >= 18) | (df['hour'] < 6)

overall_rate = df['search_conducted'].mean()
tmp = df.dropna(subset=['svi_rpl_themes', 'search_conducted']).copy()
tmp['search_01'] = tmp['search_conducted'].astype(int)
tmp['svi_quartile'] = pd.qcut(tmp['svi_rpl_themes'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])

svi_summary = tmp.groupby('svi_quartile', observed=True).agg(
    n=('search_01', 'size'),
    search_rate=('search_01', 'mean')
).reset_index()
svi_summary['search_rate_pct'] = (svi_summary['search_rate'] * 100).round(2)
svi_summary['ratio_to_baseline'] = (svi_summary['search_rate'] / overall_rate).round(2)

race_summary = df.groupby('subject_race').agg(
    n=('search_conducted', 'size'),
    search_rate=('search_conducted', lambda x: x.eq(True).mean())
).reset_index()
race_summary['search_rate_pct'] = (race_summary['search_rate'] * 100).round(2)
race_summary = race_summary.sort_values('search_rate', ascending=False)

top_races = tmp['subject_race'].value_counts().head(5).index.tolist()
tmp_race = tmp[tmp['subject_race'].isin(top_races)]
race_svi = tmp_race.groupby(['subject_race', 'svi_quartile'], observed=True)['search_01'].mean().unstack()
race_svi_pct = (race_svi * 100).round(2)

n_total = len(df)
n_searched = int(df['search_conducted'].sum())
overall_rate_pct = overall_rate * 100


def wrap_text(text, width=90):
    """Simple word wrap for long text."""
    words = text.split()
    lines = []
    current = []
    for w in words:
        if len(' '.join(current + [w])) <= width:
            current.append(w)
        else:
            if current:
                lines.append(' '.join(current))
            current = [w]
    if current:
        lines.append(' '.join(current))
    return lines


def add_text_page(pdf, title, body_paragraphs, fontsize_title=16, fontsize_body=10):
    """Add a page with title and body text."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, title, ha='center', fontsize=fontsize_title, fontweight='bold')
    y = 0.88
    for para in body_paragraphs:
        for line in wrap_text(para):
            fig.text(0.1, y, line, ha='left', va='top', fontsize=fontsize_body)
            y -= 0.04
        y -= 0.02  # extra space between paragraphs
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def main():
    with PdfPages('EDA Report.pdf') as pdf:
        # --- Title page ---
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.6, 'EDA Report', ha='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.5, 'San Diego Police Stops + Social Vulnerability Index', ha='center', fontsize=14)
        fig.text(0.5, 0.4, 'Exploratory Data Analysis', ha='center', fontsize=12)
        fig.text(0.5, 0.2, f'Data: {n_total:,} stops (2014–2017)', ha='center', fontsize=10)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # --- 1. Executive Summary ---
        add_text_page(pdf, '1. Executive Summary', [
            'This report summarizes key findings from exploratory data analysis of San Diego police traffic and '
            'pedestrian stops merged with Census tract-level Social Vulnerability Index (SVI) data.',
            f'The dataset contains {n_total:,} stops from 2014–2017. Of these, {n_searched:,} involved a search '
            f'(overall search rate: {overall_rate_pct:.2f}%).',
            'Key findings: (1) Search rates are ~2.3× higher in high-vulnerability areas (Q4) than low (Q1). '
            '(2) Black subjects have the highest search rate across all SVI quartiles. (3) Stop context—reason for '
            'stop and time of day—strongly affects search likelihood. (4) Neighborhood vulnerability is associated '
            'with search rates independent of demographics.'
        ])

        # --- 2. Target Variable Summary ---
        add_text_page(pdf, '2. Target Variable Summary', [
            f'Total stops: {n_total:,}',
            f'Stops with search: {n_searched:,}',
            f'Overall search rate: {overall_rate_pct:.2f}%',
            '',
            'By year: 2014 had the highest search rate (4.92%); 2015 and 2016 were lower (~3.8%); 2017 data '
            'covers only Jan–Mar with a rate of 4.38%. The year-over-year variation may reflect policy changes, '
            'reporting differences, or shifts in stop composition.'
        ])

        # --- 3. Search Rate by SVI Quartile (table + analysis) ---
        svi_text = 'Search rate by SVI quartile: ' + '; '.join(
            [f"{r['svi_quartile']}: {r['search_rate_pct']}%" for _, r in svi_summary.iterrows()])
        add_text_page(pdf, '3. Search Rate by Place (SVI Quartile)', [
            'The Social Vulnerability Index measures census-tract vulnerability. Quartiles: Q1 = lowest vulnerability, '
            'Q4 = highest.',
            svi_text,
            'Analysis: Search rates rise sharply with vulnerability. Q4 areas have ~2.3× the search rate of Q1. '
            'This suggests either (a) more suspicion-generating activity in high-vulnerability areas, (b) different '
            'officer behavior or deployment, or (c) both. The pattern holds across racial groups.'
        ])

        # --- 4. Search Rate by Race ---
        race_text = 'Top groups: ' + '; '.join(
            [f"{r['subject_race']}: {r['search_rate_pct']}%" for _, r in race_summary.head(5).iterrows()])
        add_text_page(pdf, '4. Search Rate by Race', [
            race_text,
            'Analysis: Black subjects have the highest search rate (9.07%), over 3× the rate for white and '
            'Asian/Pacific Islander subjects. Hispanic subjects have an intermediate rate (5.55%). These disparities '
            'persist within each SVI quartile—Black subjects have the highest rate in every quartile.'
        ])

        # --- 5. Graph 1: SVI Quartile ---
        overall_rate_pct_val = overall_rate * 100
        ratio = svi_summary['ratio_to_baseline'].values
        labels = svi_summary['svi_quartile'].astype(str).tolist()
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(x, ratio, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
        ax.axhline(1, color='gray', linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Ratio to Overall Baseline')
        ax.set_title('Graph 1: Search Rate by SVI Quartile (Ratio to Baseline)')
        for i, (b, v) in enumerate(zip(bars, ratio)):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f'{v:.2f}×', ha='center', fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # --- 6. Graph 2: Race ---
        race_plot = race_summary.head(6)
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(race_plot))
        bars = ax.bar(x, race_plot['search_rate_pct'], color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(race_plot['subject_race'], rotation=30, ha='right')
        ax.set_ylabel('Search Rate (%)')
        ax.set_title('Graph 2: Search Rate by Race')
        ax.axhline(overall_rate_pct_val, color='coral', linestyle='--', label=f'Overall ({overall_rate_pct_val:.1f}%)')
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # --- 7. Graph 3: Reason for Stop ---
        top_reasons = df['reason_for_stop'].value_counts().head(8).index.tolist()
        reason_df = df[df['reason_for_stop'].isin(top_reasons)]
        reason_summary = reason_df.groupby('reason_for_stop').agg(
            search_rate=('search_conducted', lambda x: x.eq(True).mean())
        ).reset_index()
        reason_summary = reason_summary.sort_values('search_rate', ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(reason_summary))
        ax.barh(x, reason_summary['search_rate'] * 100, color='steelblue')
        ax.set_yticks(x)
        ax.set_yticklabels(reason_summary['reason_for_stop'], fontsize=9)
        ax.set_xlabel('Search Rate (%)')
        ax.set_title('Graph 3: Search Rate by Reason for Stop')
        ax.axvline(overall_rate_pct_val, color='coral', linestyle='--', alpha=0.8)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # --- 8. Graph 4: Day vs Night ---
        daynight = df.groupby('is_night').agg(
            search_rate=('search_conducted', lambda x: x.eq(True).mean())
        ).reset_index()
        daynight['label'] = daynight['is_night'].map({True: 'Night (6pm–6am)', False: 'Day'})
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(daynight['label'], daynight['search_rate'] * 100, color=['steelblue', 'coral'])
        ax.set_ylabel('Search Rate (%)')
        ax.set_title('Graph 4: Search Rate by Day vs Night')
        ax.axhline(overall_rate_pct_val, color='gray', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # --- 9. Graph 5: Monthly Stops ---
        monthly = df.groupby([df['date'].dt.year.rename('year'), df['date'].dt.month.rename('month')]).size().reset_index(name='n_stops')
        monthly_pivot = monthly.pivot(index='month', columns='year', values='n_stops')
        monthly_search = df.groupby([df['date'].dt.year.rename('year'), df['date'].dt.month.rename('month')]).agg(
            search_rate=('search_conducted', lambda x: x.eq(True).mean())
        ).reset_index()
        search_pivot = monthly_search.pivot(index='month', columns='year', values='search_rate') * 100

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        monthly_pivot.plot(ax=axes[0], marker='o', markersize=4)
        axes[0].set_title('Number of Stops per Month (by Year)')
        axes[0].set_ylabel('Stops')
        axes[0].set_xlabel('Month')
        axes[0].set_xticks(range(1, 13))
        axes[0].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        axes[0].legend(title='Year')
        search_pivot.plot(ax=axes[1], marker='o', markersize=4)
        axes[1].set_title('Search Rate per Month (by Year)')
        axes[1].set_ylabel('Search Rate (%)')
        axes[1].set_xlabel('Month')
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        axes[1].legend(title='Year')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # --- 10. Summary ---
        add_text_page(pdf, '5. Summary & Conclusions', [
            '1. SVI: Search rates are ~2.3× higher in high-vulnerability areas (Q4) than low (Q1).',
            '2. Race: Black subjects have the highest search rate; disparities persist across SVI quartiles.',
            '3. Context: Search rates rise for non-traffic stops (e.g., Radio Call) vs routine traffic; night stops '
            'have higher search rates than day stops.',
            '4. Volume: Monthly stop counts show seasonal variation; 2017 data is partial (Jan–Mar only).',
            '',
            'These patterns suggest that both place (neighborhood vulnerability) and subject demographics are '
            'associated with search likelihood. Further causal analysis would require controlling for confounders '
            'and considering policy implications.'
        ])

    print('Report saved to EDA Report.pdf')


if __name__ == '__main__':
    main()
