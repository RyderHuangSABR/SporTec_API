def recommend_arsenal(
    target_features: np.ndarray,
    nn_model: NearestNeighbors,
    weights: np.ndarray,
    df: pd.DataFrame,
    pitcher_id_col: str = "pitcher",
    pitch_type_col: str = "pitch_type"
) -> dict:
    """
    Recommends a pitch arsenal based on the nearest historical pitch clone.
    
    Returns:
        {
            "clone_pitch": pd.Series,
            "distance": float,
            "arsenal": pd.DataFrame
        }
    """
    logger.info("Generating arsenal recommendation via 1-NN clone...")
    
    # Step 1: Find nearest neighbor (clone)
    clone_pitch, distance = get_historical_clone(
        target_features, nn_model, weights, df
    )
    
    # Step 2: Identify the pitcher of the clone
    clone_pitcher_id = clone_pitch[pitcher_id_col]
    
    # Step 3: Get all pitches from that pitcher
    pitcher_df = df[df[pitcher_id_col] == clone_pitcher_id].copy()
    
    if pitcher_df.empty:
        logger.warning("No pitches found for clone pitcher.")
        return {
            "clone_pitch": clone_pitch,
            "distance": distance,
            "arsenal": pd.DataFrame()
        }
    
    # Step 4: Build arsenal (pitch mix)
    arsenal = (
        pitcher_df[pitch_type_col]
        .value_counts(normalize=True)
        .reset_index()
    )
    
    arsenal.columns = ["pitch_type", "usage"]
    
    # Optional: map to pitch groups
    if "pitch_group" in pitcher_df.columns:
        group_arsenal = (
            pitcher_df["pitch_group"]
            .value_counts(normalize=True)
            .reset_index()
        )
        group_arsenal.columns = ["pitch_group", "usage"]
    else:
        group_arsenal = None
    
    logger.info(f"Arsenal generated for pitcher {clone_pitcher_id}")
    
    return {
        "clone_pitch": clone_pitch,
        "distance": distance,
        "arsenal": arsenal,
        "group_arsenal": group_arsenal
    }
