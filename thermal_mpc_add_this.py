# After solving optimization, before returning results
if T_pred is not None:
    # Log the prediction and current state for validation
    import os
    import json
    from datetime import datetime
    
    timestamp = datetime.now().isoformat()
    prediction_log = {
        "timestamp": timestamp,
        "current_state": x_current.tolist() if x_current is not None else None,
        "predicted_trajectories": T_pred.tolist() if T_pred is not None else None,
        "ac_decision": ac_first,
        "damper_decisions": damper_first,
        "ambient_temp": future_amb[0] if future_amb else None
    }
    
    # Save to file (ensure directory exists)
    os.makedirs("data/mpc_predictions", exist_ok=True)
    with open(f"data/mpc_predictions/pred_{timestamp.replace(':', '-')}.json", "w") as f:
        json.dump(prediction_log, f, indent=2)
