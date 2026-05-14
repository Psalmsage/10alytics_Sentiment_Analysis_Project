import os
from transformers import TrainingArguments

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(BASE_DIR)

input_data=os.path.join(BASE_DIR, "Data","testing_data.csv")

Clean_data=os.path.join(BASE_DIR, "Data","cleaned_reviews.csv")
train_data=os.path.join(BASE_DIR, "Data","processed_data","train_data.pt")
test_data=os.path.join(BASE_DIR, "Data","processed_data","test_data.pt")


model_name = "distilbert-base-multilingual-cased"
number_of_labels = 3
registered_model_name = "sentiment_model"
training_args = TrainingArguments(
    output_dir = "./results",
    num_train_epochs = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 32,
    eval_strategy= "epoch",
    save_strategy = "epoch",
    logging_dir = "./logs",
    logging_steps = 50,
    save_total_limit = 1,
    load_best_model_at_end= True,
    metric_for_best_model = "accuracy"
)