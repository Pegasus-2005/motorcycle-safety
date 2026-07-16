import random
import os

def zero_ram_split():
    print("="*60)
    print(" PHASE 3: TRANSFER LEARNING (DOMAIN ADAPTATION) ")
    print(" Executing 10/90 Split (Zero-RAM Streaming Mode)")
    print("="*60)

    dataset_path = 'boubezoul_matrix.csv' 
    if not os.path.exists(dataset_path):
        dataset_path = 'data/boubezoul_matrix.csv'
        if not os.path.exists(dataset_path):
            print(f"[ERROR] Could not find Boubezoul matrix. Are you in the right folder?")
            return

    train_path = 'boubezoul_transfer_train.csv'
    test_path = 'boubezoul_transfer_test.csv'

    train_count = 0
    test_count = 0

    print(f"[SYSTEM] Streaming data line-by-line to bypass Mac RAM limits...")
    print(f"[SYSTEM] Please wait roughly 5 to 15 seconds...")
    
    with open(dataset_path, 'r', encoding='utf-8') as src, \
         open(train_path, 'w', encoding='utf-8') as trn, \
         open(test_path, 'w', encoding='utf-8') as tst:
        
        header = src.readline()
        trn.write(header)
        tst.write(header)

        for line in src:
            if line.strip() == "": continue
            
            if random.random() < 0.10:
                trn.write(line)
                train_count += 1
            else:
                tst.write(line)
                test_count += 1

    print("-" * 60)
    print(f"[SUCCESS] 10% Transfer Learning Set Created: ~{train_count} samples -> '{train_path}'")
    print(f"[SUCCESS] 90% Unseen Evaluation Set Created: ~{test_count} samples -> '{test_path}'")
    print("="*60)

if __name__ == "__main__":
    zero_ram_split()