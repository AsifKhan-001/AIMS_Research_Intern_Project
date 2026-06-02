import json
import os
import time  # ◀── 1. Add this import at the top!
from src.agent_graph import agent_app

def execute_batch_evaluation(config_name, ablation_flag):
    """
    Processes the complete question collection using a specific system configuration
    and writes out the required predictions line-by-line.
    """
    input_path = "eval/questions.jsonl"
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{config_name}.jsonl"
    
    print(f"\n>>> Running Evaluation Matrix for Configuration: [{config_name.upper()}] <<<")
    
    questions = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line.strip()))
                
    with open(output_path, "w") as out_f:
        for item in questions:
            q_id = item["id"]
            question_text = item["question"]
            
            print(f" -> Solving Question ID: {q_id}")
            
            try:
                # Invoke the LangGraph framework using our custom ablation tracking key
                result = agent_app.invoke({
                    "question": question_text,
                    "ablation_mode": ablation_flag
                })
                
                # Structure the dictionary output to match the grading specification perfectly
                submission_row = {
                    "id": q_id,
                    "answer": result.get("verified_answer", ""),
                    "cited_papers": result.get("cited_papers", [])
                }
                
                out_f.write(json.dumps(submission_row) + "\n")
                out_f.flush() # Force write immediately to the file
                
            except Exception as e:
                print(f" ❌ Error processing {q_id}: {e}")
                print("Retrying after 60 seconds...")
                time.sleep(60)
                try:
                    result = agent_app.invoke({
                        "question": question_text,
                        "ablation_mode": ablation_flag
                    })
                    submission_row = {
                        "id": q_id,
                        "answer": result.get("verified_answer", ""),
                        "cited_papers": result.get("cited_papers", [])
                    }
                    out_f.write(json.dumps(submission_row) + "\n")
                    out_f.flush()
                except Exception as e2:
                    print(f" ❌ Final failure {q_id}: {e2}")
            
            # ◀── 2. Add a 5-second sleep pause here to pace out the free tier requests!
            print("Pacing API requests... resting for 5 seconds...")
            time.sleep(6)
            
    print(f"Finished! Prediction file safely saved to: {output_path}")

if __name__ == "__main__":
    # Ensure our predictions folder exists
    os.makedirs("predictions", exist_ok=True)
    
    # Run all required ablation configurations specified in the assignment
    ablation_matrix = {
        "full_agent": "none",          # The complete, unablated agentic framework
        "baseline": "baseline",        # Naive RAG simulation
        "no_planner": "no_planner",    # Skip query decomposition
        "no_reflector": "no_reflector",# Skip loop checks
        "no_hybrid": "no_hybrid",      # Dense vector matching only
        "no_verifier": "no_verifier"   # Skip citation verification
    }
    
    for config_title, flag_key in ablation_matrix.items():
        execute_batch_evaluation(config_title, flag_key)
        
    print("\n========================================================")
    print("ALL ABLATION FILES SUCCESSFULLY COMPILED FOR GRADING!")
    print("Check the 'predictions/' folder to review your outputs.")
    print("========================================================")