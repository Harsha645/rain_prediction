# Given a date and city, looks up that day's weather from the dataset
# and predicts whether it will rain the NEXT day using both trained models.
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import FEATURE_COLS, RAIN_THRESHOLD


def predict_rain_for_date(
    input_date  : str,
    input_city  : str,
    df          : pd.DataFrame,
    dt_model,
    rf_model,
    scaler,
    feature_cols: list
):
    """
    Predict whether it will rain the day after input_date in input_city.

    The function uses TODAY'S weather values as input to predict TOMORROW'S rain.
    This is how the model was trained:
        f(today's weather) = rain tomorrow? (1 or 0)

    Parameters
    input_date   : str           Date in 'YYYY-MM-DD' format
    input_city   : str           City name (case-insensitive)
    df           : pd.DataFrame  Full preprocessed dataset
    dt_model     : Trained DecisionTreeClassifier
    rf_model     : Trained RandomForestClassifier
    scaler       : Fitted StandardScaler
    feature_cols : list          Feature columns used during training

    Returns
    None  (prints formatted prediction report)
    """

    # ── Step 1: Validate city name ────────────────────────────────────────────
    # Build case-insensitive lookup dictionary
    city_map = {c.lower(): c for c in df['city'].unique()}

    if input_city.lower() not in city_map:
        print(f'City "{input_city}" not found in dataset.')
        print(f'Available cities: {sorted(df["city"].unique())}')
        return

    matched_city = city_map[input_city.lower()]

    # ── Step 2: Parse and validate date ──────────────────────────────────────
    try:
        query_date = pd.to_datetime(input_date)
    except Exception:
        print(f'Invalid date format: "{input_date}". Use YYYY-MM-DD.')
        return

    next_date = query_date + pd.Timedelta(days=1)

    # ── Step 3: Look up weather row for (city, date) ─────────────────────────
    # This retrieves TODAY'S weather values — the model's input
    row = df[(df['city'] == matched_city) & (df['time'] == query_date)]

    if row.empty:
        # Tell user what date range is available for this city
        city_df = df[df['city'] == matched_city]
        min_d   = city_df['time'].min().strftime('%Y-%m-%d')
        max_d   = city_df['time'].max().strftime('%Y-%m-%d')
        print(f'No data found for {matched_city} on {input_date}.')
        print(f'Available date range for {matched_city}: {min_d} to {max_d}')
        return

    # ── Step 4: Extract and scale feature values ──────────────────────────────
    # Fill any missing values with dataset median (same as training)
    X_input        = row[feature_cols].fillna(
                        df[feature_cols].median())
    X_input_scaled = scaler.transform(X_input)

    # ── Step 5: Run predictions through both models ───────────────────────────
    dt_result = dt_model.predict(X_input_scaled)[0]
    dt_prob   = dt_model.predict_proba(X_input_scaled)[0]

    rf_result = rf_model.predict(X_input_scaled)[0]
    rf_prob   = rf_model.predict_proba(X_input_scaled)[0]

    dt_label  = 'RAIN' if dt_result == 1 else 'NO RAIN'
    rf_label  = 'RAIN' if rf_result == 1 else 'NO RAIN'

    # ── Step 6: Check actual outcome (if available in dataset) ───────────────
    actual_row = df[(df['city'] == matched_city) & (df['time'] == next_date)]

    if not actual_row.empty:
        actual_precip = actual_row['precipitation_sum'].values[0]
        actual_label  = 'RAIN' if actual_precip > RAIN_THRESHOLD else 'NO RAIN'
        actual_str    = f'{actual_label}  (actual precipitation: {actual_precip:.1f} mm)'
    else:
        actual_str    = 'Not available (date is outside dataset range)'

    # ── Step 7: Print formatted prediction report ─────────────────────────────
    line = '=' * 54
    print(line)
    print('  Rain Prediction Report')
    print(line)
    print(f'  City          : {matched_city}')
    print(f'  Query Date    : {query_date.strftime("%d %B %Y")}  (input weather)')
    print(f'  Predict For   : {next_date.strftime("%d %B %Y")}  (next day)')
    print()
    print(f"  Today's Weather ({query_date.strftime('%d %b %Y')}):")
    print(f'    Temperature (mean) : {row["temperature_2m_mean"].values[0]:.1f} C')
    print(f'    Precipitation      : {row["precipitation_sum"].values[0]:.1f} mm')
    print(f'    Wind Speed (max)   : {row["windspeed_10m_max"].values[0]:.1f} km/h')
    print(f'    Solar Radiation    : {row["shortwave_radiation_sum"].values[0]:.1f} MJ/m2')
    print()
    print(f'  Decision Tree  ->  {dt_label}')
    print(f'    No Rain: {dt_prob[0]*100:.1f}%   Rain: {dt_prob[1]*100:.1f}%')
    print()
    print(f'  Random Forest  ->  {rf_label}')
    print(f'    No Rain: {rf_prob[0]*100:.1f}%   Rain: {rf_prob[1]*100:.1f}%')
    print()
    print(f'  Actual Result  ->  {actual_str}')
    print(line)


def plot_prediction_chart(
    input_date  : str,
    input_city  : str,
    df          : pd.DataFrame,
    dt_model,
    rf_model,
    scaler,
    feature_cols: list
):
    """
    Bar chart showing Rain probability from both models for a given date/city.

    Provides a visual confirmation of the prediction with probability percentages
    shown on each bar. The actual outcome is shown in the x-axis label.

    Parameters
    Same as predict_rain_for_date()
    """
    # Validate city
    city_map = {c.lower(): c for c in df['city'].unique()}
    if input_city.lower() not in city_map:
        print(f'City not found: {input_city}')
        return

    matched_city = city_map[input_city.lower()]
    query_date   = pd.to_datetime(input_date)
    next_date    = query_date + pd.Timedelta(days=1)

    # Look up row
    row = df[(df['city'] == matched_city) & (df['time'] == query_date)]
    if row.empty:
        print(f'No data for {matched_city} on {input_date}.')
        return

    # Scale and predict
    X_in  = row[feature_cols].fillna(df[feature_cols].median())
    X_sc  = scaler.transform(X_in)
    dt_p  = dt_model.predict_proba(X_sc)[0]
    rf_p  = rf_model.predict_proba(X_sc)[0]

    # Get actual outcome
    actual_row = df[(df['city'] == matched_city) & (df['time'] == next_date)]
    actual_lbl = None
    if not actual_row.empty:
        actual_lbl = 'Rain' if actual_row['precipitation_sum'].values[0] \
                     > RAIN_THRESHOLD else 'No Rain'

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, title, prob, colors in zip(
        axes,
        ['Decision Tree', 'Random Forest'],
        [dt_p, rf_p],
        [['#4A90D9', '#E74C3C'], ['#27AE60', '#E74C3C']]
    ):
        bars = ax.bar(['No Rain', 'Rain'], prob * 100,
                      color=colors, edgecolor='white', linewidth=1.5)

        # Add percentage labels on bars
        for bar, val in zip(bars, prob * 100):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold'
            )

        pred_lbl = 'Rain' if prob[1] >= 0.5 else 'No Rain'
        ax.set_ylim(0, 115)
        ax.set_ylabel('Probability (%)')
        ax.set_title(
            f'{title}\n{matched_city} | {query_date.strftime("%d %b %Y")}',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel(
            'Prediction: ' + pred_lbl +
            (f'  |  Actual: {actual_lbl}' if actual_lbl else ''),
            fontsize=11
        )
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        f'Rain Probability for {next_date.strftime("%d %B %Y")} (next day)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.show()
