def merge_data(health_df, weather_df, water_df, geo_df):
    latest_health = (
        health_df.groupby("village_id")["reported_cases"]
        .sum()
        .reset_index()
    )

    latest_data = latest_health.merge(geo_df, on="village_id")

    latest_data = latest_data.merge(weather_df, on="village_id", how="left")

    water_features = water_df[["ph", "Turbidity", "Conductivity"]].reset_index(drop=True)
    for col in water_features.columns:
        latest_data[col] = (
            water_features[col]
            .sample(n=len(latest_data), replace=True, random_state=42)
            .values
        )

    return latest_data
