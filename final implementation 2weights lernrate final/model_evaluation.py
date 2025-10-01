# model_evaluation.py
import torch
from sklearn.metrics import classification_report

def evaluate_model(model, data_loader, action_label_encoder):
    """
    Evaluates the model on a given dataset (e.g., validation set).
    Computes and prints a classification report for the action prediction task.
    Returns the macro F1-score.
    """
    device = next(model.parameters()).device
    model.eval()
    
    all_true_actions, all_pred_actions = [], []
    
    with torch.no_grad():
        for xc, xcont, x360, yz, ya, yt in data_loader:
            xc, xcont, x360 = xc.to(device), xcont.to(device), x360.to(device)
            
            _, action_logits, _ = model(xc, xcont, x360)
            
            all_true_actions.extend(ya.numpy())
            all_pred_actions.extend(action_logits.argmax(dim=-1).cpu().numpy())

    print("\n--- Trial Evaluation Results ---")
    target_names = [action_label_encoder[i] for i in sorted(action_label_encoder.keys())]
    
    # Generate and print the classification report
    report_str = classification_report(
        all_true_actions, 
        all_pred_actions, 
        target_names=target_names, 
        zero_division=0, 
        digits=2
    )
    print(report_str)
    
    # Extract the macro F1-score to return to Optuna
    report_dict = classification_report(
        all_true_actions, 
        all_pred_actions, 
        target_names=target_names, 
        zero_division=0, 
        output_dict=True
    )
    macro_f1 = report_dict['macro avg']['f1-score']
    print(f"===> Macro F1-Score for this trial: {macro_f1:.4f}")
    
    return macro_f1