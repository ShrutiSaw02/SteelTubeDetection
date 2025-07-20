import os

label_dirs = ['dataset/labels/train', 'dataset/labels/val']

for label_dir in label_dirs:
    for fname in os.listdir(label_dir):
        if fname.endswith('.txt'):
            path = os.path.join(label_dir, fname)
            with open(path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    parts[0] = '0'  # Set class ID to 0
                    new_lines.append(' '.join(parts))

            with open(path, 'w') as f:
                f.write('\n'.join(new_lines))

print(" All label class IDs updated to 0.")