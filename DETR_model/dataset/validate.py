import os

# Check if labels are normalized (0-1 range)
label_dir = 'labels'
problems = []

for label_file in os.listdir(label_dir):
    if not label_file.endswith('.txt'):
        continue
    
    with open(os.path.join(label_dir, label_file)) as f:
        for line_num, line in enumerate(f, 1):
            vals = list(map(float, line.strip().split()))
            if len(vals) != 5:
                problems.append(f"{label_file} line {line_num}: Expected 5 values, got {len(vals)}")
                continue
            
            class_id, cx, cy, w, h = vals
            
            # Check if normalized (0-1)
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                problems.append(f"{label_file} line {line_num}: Values not normalized! cx={cx} cy={cy} w={w} h={h}")

if problems:
    print("❌ PROBLEMS FOUND:")
    for p in problems[:10]:  # Show first 10
        print(f"  - {p}")
else:
    print("✅ All labels look good!")