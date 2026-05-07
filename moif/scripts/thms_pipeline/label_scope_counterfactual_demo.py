import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

def run_counterfactual_demo():
    print("--- LabelScope Counterfactual Demonstration ---")
    
    # 1. Setup: Pseudo-replicated Data (Scenario: Block-level state, Minute-level physiology)
    n_blocks = 10
    samples_per_block = 100
    n_samples = n_blocks * samples_per_block
    
    # Physiological features
    X = np.random.normal(0, 1, (n_samples, 2))
    
    # Block-level labels (The "True" state only changes every 100 samples)
    block_labels = np.random.randint(0, 2, n_blocks)
    y_pseudo = np.repeat(block_labels, samples_per_block)
    
    # 2. Naive Pipeline (WITHOUT LabelScope)
    # The researcher treats all minute-level samples as independent
    X_train, X_test, y_train, y_test = train_test_split(X, y_pseudo, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    naive_acc = accuracy_score(y_test, y_pred)
    
    print(f"\n[Naive Result] Minute-level Classification Accuracy: {naive_acc:.3f}")
    print(">> Researcher Claim: 'Our system can monitor this state with ~90% accuracy.'")
    
    # 3. WITH LabelScope (Audit Phase)
    print("\n[LabelScope Audit Execution]")
    
    # Resolution Audit
    n_independent = n_blocks
    pseudo_rep_factor = n_samples / n_independent
    
    ra_fail = n_independent < 100 and pseudo_rep_factor > 10
    
    print(f"- Independent Label Count (N_ind): {n_independent}")
    print(f"- Pseudo-replication Factor (R_P): {pseudo_rep_factor:.1f}")
    
    if ra_fail:
        print(">> VERDICT: CLAIM LEVEL 0 (Invalid Candidate)")
        print(">> RISK: Accuracy is inflated by pseudo-replication artifacts.")
        print(">> CORRECTED CLAIM: 'Accuracy reflects temporal dependency, not independent state modeling.'")

    # 4. Save results for dashboard
    report = {
        "naive_pipeline": {"accuracy": naive_acc, "claim": "Minute-level monitoring possible"},
        "label_scope_audit": {
            "n_ind": n_independent,
            "rp": pseudo_rep_factor,
            "verdict": "Level 0",
            "risk": "Pseudo-replication artifact detected"
        }
    }
    with open("label_scope_counterfactual_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    run_counterfactual_demo()
