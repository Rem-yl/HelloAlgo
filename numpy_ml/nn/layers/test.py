sd = {
    "parameters": {"lr": 0.01, "weight_decay": 0.001},  # 嵌套字典1
    "hyperparameters": {"batch_size": 32, "epochs": 100},  # 嵌套字典2
    "model_name": "ResNet50"  # 顶层键
}

flatten_keys = ["parameters", "hyperparameters"]
for k in flatten_keys:
    if k in sd:
        entry = sd[k]
        sd.update(entry)
        del sd[k]
print(sd)
