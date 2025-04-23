import re
import matplotlib.pyplot as plt

# Load your log file
log_path = 'nohup.out'
with open(log_path, 'r') as f:
    lines = f.readlines()

# Prepare lists to store extracted data
iterations = []
train_losses = []
val_iterations = []
val_losses = []

for line in lines:
    # Match iteration/training loss lines
    match_train = re.search(r'Iteration\s*\[\s*(\d+)/\d+\]\s*\|\s*loss:\s*([0-9.]+)', line)
    if match_train:
        iter_num = int(match_train.group(1))
        loss_val = float(match_train.group(2))
        iterations.append(iter_num)
        train_losses.append(loss_val)
        continue
    
    # Match validation loss lines
    match_val = re.search(r'Validation\s*\|\s*loss:\s*([0-9.]+)', line)
    if match_val:
        loss_val = float(match_val.group(1))
        # (Optional) Associate validation with last known iteration
        val_iterations.append(iterations[-1] if iterations else 0)
        val_losses.append(loss_val)

# Now you have iterations/train_losses and val_iterations/val_losses

# Plot it
plt.figure(figsize=(10,6))
plt.plot(iterations, train_losses, label='Training Loss')
plt.plot(val_iterations, val_losses, 'o-', label='Validation Loss')  # 'o-' means dots + lines
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()
