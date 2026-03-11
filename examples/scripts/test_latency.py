import time

from agent_action_guard import is_action_harmful
from training.train_nn import dataset

if not dataset:
    raise ValueError("Dataset is empty. Please ensure the dataset is loaded correctly.")


def main():
    start_time = time.time()
    for row in dataset:
        _ = is_action_harmful(row["action"])

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Processed {len(dataset)} actions.")

    avg_time_per_action = (end_time - start_time) / len(dataset)
    avg_time_ms = avg_time_per_action * 1000.0
    print(f"Average time per action: {avg_time_ms:.4f} ms")


if __name__ == "__main__":
    main()
