from train_utils import *


model_type = 'awf'
dataset = "sirinam"
num_classes = str(95)
config = ConfigObj("My_tor.conf")
learn_param = config[model_type]
model = build_model_instance(model_type, dataset+num_classes, config).to(device)
model.load_state_dict(torch.load(f"../utils/trained_model/{dataset}{num_classes}/{model_type}.pkl"))
train_loader, val_loader, test_loader  = load_datasets(dataset, int(num_classes), learn_param)
# acc, recall, precision, f1 = test_metrics(model, test_loader)
# print(f"Test Accuracy: {acc:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, F1-score: {f1:.3f}")
analyze_model_performance(model, test_loader, dataset, int(num_classes), model_type, 
                              train_time=None, device='cuda', log_path=None, 
                              warmup=10, repeat=100)