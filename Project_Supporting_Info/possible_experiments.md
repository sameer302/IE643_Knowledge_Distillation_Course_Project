1) comparing results with and without freezing the teacher backbone while finetuning
2) comparing results by changing the alpha value in KD script
3) sequential data KD and combining data KD
4) In sequential KD, we can change alpha value for each dataset by visualising whether teacher model is predicting close to actual predictions or not 

1) Understand the mapping of features
2) Why the total parameters changed for student ?
3) Novelty try adaptive alpha and feature level cached predictions.
4) feature projection instead of padding and truncation in multi_loader


<!-- Date-  Nov 7
🧩 Summary — What We Did and Why
Problem:
    Each dataset originally had a different number of features (some smaller, some larger than 42).
    The old multi_loader.py padded or truncated each dataset to match the teacher’s 42-feature dimension.
    → This made batching possible, but caused information loss (especially for large datasets like Household with 321 features) and made projection adapters useless.

Solution – multi_loader_new.py:
    We redesigned the loader to:
        Pad every dataset to the maximum feature count (e.g., 321) — no truncation at all.
        Record each dataset’s original feature dimension (orig_dim) so that information isn’t lost.
        Still concatenate all datasets into one unified dataloader for global training.
    ✅ This preserved all features and kept batching safe.
Solution – run_finetune_global_new.py:
    We updated the fine-tune script to use:
        A new ProjectedTeacherNew model with per-dataset projection layers (Linear(orig_dim → 42) and Linear(42 → orig_dim)).
        Each dataset now learns its own adapter to map between its feature space and the teacher’s latent 42-dim space.
        The teacher backbone stays frozen (unless unfrozen explicitly).
    ✅ This allows proper learnable projection instead of static padding/truncation.

Outcome:
    No information is lost from any dataset.
    All datasets share one global teacher backbone but have dataset-specific projection adapters.
    The system can now perform true global fine-tuning across heterogeneous datasets, safely and efficiently. -->

    1) Seq teacher+seq_student
    2) Global teacher+global-student_const_alpha (old multiloader)
    3) Global teacher+global-student_dict_varying_alpha (old multiloader)

    4) Global Teacher + global-student_const_alpha (new-multi-loader)
    5) Global Teacher + global-adaptive_alpha 
    6) Global Teacher + global-learning-rate_alpha 
    7) Global Teacher + global-adaptive+smoothing_alpha 

    8) Global teacher + global-adaptive+smoothing_alpha (Feature_cached-preds)