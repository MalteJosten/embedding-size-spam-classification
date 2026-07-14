import os

with open("mails/spam_train.txt", "r") as train_file:
    train_set = []
    for line in train_file.readlines():
        if line:
            train_set.append(line)

with open("mails/spam_test.txt", "r") as test_file:
    test_set = []
    for line in test_file.readlines():
        if line:
            test_set.append(line)

print(f"#Train: {len(train_set)}, #Test: {len(test_set)}")

found = 0
for item in train_set:
    if item in test_set:
        found += 1
 
print(f"Found {found} duplicates!")