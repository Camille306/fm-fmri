# Data Splitting Strategy - No Data Leakage

## Answer: **NO, the same subject will NOT appear in different splits**

The dataset splitting is done at the **subject level**, not at the window level. This ensures proper evaluation without data leakage.

## How It Works

### Subject-Level Splitting

Looking at the code in `train.py` (lines 108-122):

```python
def _create_window_indices(self, train_ratio: float, val_ratio: float):
    all_subjects = self.dataset.subject_ids  # All subjects
    num_subjects = len(all_subjects)
    
    # Split subjects (not windows!)
    train_end = int(num_subjects * train_ratio)      # 70% of subjects
    val_end = train_end + int(num_subjects * val_ratio)  # 70% + 15% = 85%
    
    if self.split == 'train':
        subject_ids = all_subjects[:train_end]           # First 70% of subjects
    elif self.split == 'val':
        subject_ids = all_subjects[train_end:val_end]    # Next 15% of subjects
    else:  # test
        subject_ids = all_subjects[val_end:]             # Last 15% of subjects
    
    # Then create windows ONLY from subjects in this split
    for subject_id in subject_ids:
        # Create all windows for this subject
        # All windows from this subject go to the same split
```

### Example

Suppose you have 100 subjects:

1. **Subject-level split**:
   - Training: Subjects 0-69 (70 subjects)
   - Validation: Subjects 70-84 (15 subjects)
   - Testing: Subjects 85-99 (15 subjects)

2. **Window creation**:
   - For Subject 50 (in training): ALL windows from Subject 50 → Training set
   - For Subject 75 (in validation): ALL windows from Subject 75 → Validation set
   - For Subject 90 (in testing): ALL windows from Subject 90 → Testing set

### Visual Example

```
Subject 1: [Window 1, Window 2, Window 3, ..., Window N] → Training
Subject 2: [Window 1, Window 2, Window 3, ..., Window N] → Training
...
Subject 70: [Window 1, Window 2, Window 3, ..., Window N] → Training
─────────────────────────────────────────────────────────────
Subject 71: [Window 1, Window 2, Window 3, ..., Window N] → Validation
...
Subject 85: [Window 1, Window 2, Window 3, ..., Window N] → Validation
─────────────────────────────────────────────────────────────
Subject 86: [Window 1, Window 2, Window 3, ..., Window N] → Testing
...
Subject 100: [Window 1, Window 2, Window 3, ..., Window N] → Testing
```

## Why This Matters

### ✅ Correct Approach (Subject-Level Split)
- **No data leakage**: Model never sees test subjects during training
- **Proper evaluation**: Test performance reflects true generalization
- **Realistic scenario**: Simulates predicting task data for new, unseen subjects

### ❌ Wrong Approach (Window-Level Split)
If windows from the same subject could appear in different splits:
- **Data leakage**: Model could memorize subject-specific patterns
- **Overestimated performance**: Test results would be artificially high
- **Not generalizable**: Model might not work well on truly new subjects

## Code Verification

You can verify this by checking the subject IDs in each split:

```python
# After creating datasets
train_subjects = set([meta['subject_id'] for meta in train_dataset.window_metadata])
val_subjects = set([meta['subject_id'] for meta in val_dataset.window_metadata])
test_subjects = set([meta['subject_id'] for meta in test_dataset.window_metadata])

# Verify no overlap
assert len(train_subjects & val_subjects) == 0, "Subject overlap between train and val!"
assert len(train_subjects & test_subjects) == 0, "Subject overlap between train and test!"
assert len(val_subjects & test_subjects) == 0, "Subject overlap between val and test!"

print("✓ No subject overlap - proper split!")
```

## Summary

- **Splitting level**: Subject-level (not window-level)
- **Result**: Each subject's windows go to exactly one split
- **Data leakage**: None - proper evaluation setup
- **Default ratios**: 70% train, 15% val, 15% test (of subjects)

This is the **correct** approach for fMRI data where subject-specific patterns are important and we want to evaluate generalization to new subjects.
