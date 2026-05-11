# **SAE-Based Shortcut Detection**

## **and Bias Mitigation in Brain Tumor MRI Classification**

##### _A Complete Beginner's Guide to Your Research Project_

Topics Covered: Full Pipeline • Shortcut Detection • Ablation Studies • Mitigation Strategies


### **1. The Big Picture — What Is This Project About?**

Before looking at any code, let us understand the entire project using a simple analogy from real life.

#### **1.1 The Cheating Student Analogy**


In your project, the "cheating student" is an AI model trained on brain MRI scans. Instead of learning to truly
recognize tumors, the model learns a shortcut — an irrelevant image feature like the darkness of the image
border — that accidentally correlates with whether a tumor is present in the training data.

#### **1.2 Why This Happens (The Hospital Problem)**


Different hospitals use different MRI machines. Hospital A's machine happens to produce darker-bordered
images mostly for no-tumor patients, while Hospital B's machine does the opposite. If your AI model is only
trained at Hospital A, it learns:


  - "Dark border" means probably no tumor (Hospital A pattern)

  - "Bright border" means probably tumor (Hospital A pattern)


When deployed at Hospital B (with different equipment), the model fails catastrophically — because the
"dark border shortcut" no longer holds. This is called distribution shift and is a major safety problem in
medical AI.

#### **1.3 Your Project's Three Goals**


  - **1.** DETECT: Find which specific internal features of the AI model encode these shortcuts.

  - **2.** PROVE: Run experiments that scientifically prove the shortcut exists and causes failure.

  - **3.** FIX: Apply mitigation strategies that remove or reduce reliance on the shortcut.


### **2. Dataset, Splits, Gap, and Bias — Clearly Explained**

#### **2.1 The Dataset: Brain Tumor MRI**

The dataset is the Brain-Tumor-MRI-Dataset from HuggingFace. It contains thousands of brain MRI images in
two categories:


  - Tumor (label = 1): Image shows a brain tumor

  - No Tumor (label = 0): Healthy brain, no tumor


It originally has 4 tumor sub-types (glioma, meningioma, pituitary, no_tumor), but the code simplifies this to
binary: tumor vs no-tumor.

#### **2.2 What Are "Splits"?**





|Split Name|What It Contains|What It Tells You|
|---|---|---|
|Training Split|Images the model learns from<br>(80%)|Where bias enters the model|
|Test Split|Images never seen during training<br>(20%)|Measures real-world performance|
|Aligned Split (A)|Tumor+high border AND no-<br>tumor+low border|The "easy" group — shortcut<br>agrees with truth|
|Conflicting Split (B)|Tumor+low border AND no-<br>tumor+high border|The "hard" group — shortcut<br>contradicts truth|
|Shifted Test Set|Only conflicting images —<br>simulates different hospital|The critical stress-test for fairness|

#### **2.3 What Is the "Gap"?**





The GAP is the most important metric in this entire project. It measures how unfair or biased the model is.


GAP = Aligned Accuracy − Shifted Accuracy




#### **2.4 What Is "Bias" and "Cheating"?**


  - **Bias:** Bias = A statistical correlation in training data between something irrelevant (border darkness)
and the true label (tumor). This correlation is accidental — it comes from how images were
collected, not from medicine.

  - **Cheating:** Cheating = The model learns to use this accidental correlation as its prediction rule.
Instead of analyzing brain tissue, it shortcuts to: "dark border = no tumor."

  - **Spurious Correlation:** Spurious Correlation = The technical term for the accidental link. "Spurious"
means fake/irrelevant.

  - **Shortcut Feature:** Shortcut Feature = A specific internal neuron/SAE feature in the AI that has
learned this spurious correlation.


### **3. Your Full Pipeline — Step by Step**

Here is the complete pipeline from raw data to results:

#### **Step 1: Load Data and Compute Spurious Attributes**


The code loads the Brain Tumor MRI dataset and computes 8 different spurious attributes for every image

- image properties that have nothing to do with medicine but might accidentally correlate with tumor
presence:













|Attribute|What It Measures|Why Spurious|
|---|---|---|
|intensity|Average pixel brightness|Some scanners make images<br>brighter|
|edge_density|How many sharp edges exist|Different MRI protocols produce<br>different edge sharpness|
|contrast|Range of bright/dark variation|Scanner settings affect contrast|
|freq_ratio|High-frequency vs low-frequency<br>signal ratio|Different noise profiles from<br>different machines|
|dark_border|Fraction of nearly-black border<br>pixels|Some protocols add a dark<br>padding border (STRONGEST<br>BIAS)|
|skull_ratio|Bright rim pixels at image edges|Scan field-of-view differs across<br>protocols|
|texture_uniformity|Uniformity of local texture<br>patterns|Machine noise creates different<br>texture patterns|
|aspect_ratio|Original image width/height ratio|Different sources crop images<br>differently|


Each attribute is binarized by median-split: above median = "high" (1), below = "low" (0). This allows
statistical comparison between two groups.

#### **Step 2: Build the Contrast Set (Split A and Split B)**


50 images are sampled from each of the 4 (y, attribute) combinations. Then:


  - Split A (aligned): tumor+high_edge AND no-tumor+low_edge — shortcut agrees with label

  - Split B (conflicting): tumor+low_edge AND no-tumor+high_edge — shortcut contradicts label

#### **Step 3: Load CLIP + SAE (PatchSAE)**


CLIP (Contrastive Language-Image Pre-training by OpenAI) is a powerful vision model that converts images
into 512-dimensional feature vectors. It uses a Vision Transformer (ViT-B/16) backbone with 12 transformer
layers.


PatchSAE (Sparse Autoencoder) is a separate network trained ON TOP of CLIP. It decomposes CLIP's internal
768-dimensional activations into 49,152 interpretable features.


#### **Step 4: Collect SAE Activations (The Hook)**

For every contrast set image, the code uses a "forward hook" — a function that intercepts CLIP's internal
computation at Layer 10 (second-to-last layer) and captures the activations before they continue. These
activations are then:


  - Passed through the SAE to get 49,152 feature values per image patch (14×14 = 196 patches + 1 CLS
token)

  - Pooled by averaging across all patches — result: one 49,152-dimensional vector per image

#### **Step 5: Find Shortcut Features (The Core Discovery)**


For each of the 49,152 SAE features, two correlation values are computed:


  - r_spurious: Correlation between this feature's activation and the spurious attribute (e.g.,
dark_border)

  - r_label: Correlation between this feature's activation and the true tumor/no-tumor label


A feature is a SHORTCUT FEATURE if: r_spurious > r_label AND p-value < 0.05. This finds features that
encode spurious information MORE than real tumor information.

#### **Step 6: Ablate (Turn Off) Shortcut Features**




Ablation is implemented as a real-time hook: intercept activations → encode through SAE → zero out
shortcut features → decode back → continue. This all happens in real-time during inference without
changing model weights.

#### **Step 7: Measure the Effect and Report**


After ablation, the code measures: (1) Aligned Accuracy — did ablation hurt easy images? (2) Shifted
Accuracy — did ablation help hard images? (3) Gap — did fairness improve? Results are compared across 4
strategies and 6 bias ratios.


### **4. How Does the Model Cheat? — The Proof**

#### **4.1 The Probe Test**

A probe is a simple logistic regression classifier trained on CLIP's internal features to predict the spurious
attribute (not the tumor label). If the probe can predict dark_border with much more than 50% accuracy, it
means CLIP has secretly encoded dark_border information inside itself.


Result: Probe accuracy = 89%. A fair model would have 50% probe accuracy on a random attribute. Your
model has 89%, proving it has learned the dark_border shortcut.

#### **4.2 The Correlation Test (r_spurious)**


For each SAE feature, the Point-Biserial Correlation between feature activation and the spurious attribute is
computed. High r_spurious (0.6–0.7) means this feature fires specifically when the spurious attribute is
present. It is like finding the exact neuron that detects dark borders.

#### **4.3 The Accuracy Drop and Recovery Test**


The strongest proof of cheating is the accuracy pattern:


  - WITHOUT ablation: Lower accuracy on conflicting images (model is misled by the shortcut)

  - WITH ablation: Higher accuracy on conflicting images (shortcut removed, model forced to use real
features)

  - WITH RANDOM ablation (control): No improvement (proves the specific identified features matter,
not just any zeroing)

#### **4.4 The Spatial Attribution Heatmaps**


The code generates heatmaps showing WHERE each shortcut feature activates in the image. Genuine
shortcut features (encoding dark_border) should light up at image edges and corners (border regions), NOT
at the tumor in the center. These heatmaps provide visual, intuitive proof of spurious features.


### **5. The 5 Ablation Studies — Fully Explained**

"Ablation study" in ML = "what happens if we remove or change one part of our method?" It systematically
proves that your design choices matter.

#### **Ablation Study 1: Feature Count Scaling**


Question asked: Does ablating more features give better results?


Method: Test 5 different numbers of features to ablate: 0 (baseline), 5, 10, 20, and 36 (all detected
shortcuts). Measure aligned accuracy, shifted accuracy, and gap for each.


_Figure 1 — Ablation 1: Feature Count Scaling. Blue = Aligned Accuracy. Orange = Shifted Accuracy. Green bars = Gap._


How to read:


  - Blue line (Aligned Accuracy): Stays steady at ~99.3% regardless of how many features you ablate.
The model still recognizes easy tumors.

  - Orange line (Shifted Accuracy): Slightly drops from 94.0% at 0 features to 92.6–93.2% with ablation.
Small cost.

  - Green bars (Gap): Stays around 0.06 regardless. MORE FEATURES DO NOT HELP.


#### **Ablation Study 2: Per-Attribute Analysis**

Question asked: Which of the 8 spurious attributes creates the most shortcut features in CLIP?


Method: Scan all 8 attributes independently. For each, count how many of the top-50 most-active features
are shortcut features for that attribute. Also measure gap reduction when ablating those attribute-specific
features.


_Figure 2 — Ablation 2: Per-Attribute Analysis. Purple bars = shortcuts found per attribute. Red bars = gap reduction (negative_

_= gap got worse)._


How to read:


  - Purple bars (Shortcuts Found): dark_border tops the list with 27 features, followed by
texture_uniformity (24), freq_ratio (13), edge_density (21), and contrast (12). CLIP has strongly
encoded dark_border patterns.

  - Red bars (Gap Reduction): ALL values are negative (−0.005 to −0.013). Ablating per-attribute
features WORSENS the gap slightly. Why? Because those features carry mixed information — both
spurious AND real tumor signal.


#### **Ablation Study 3: Bias Ratio Sensitivity — The Key Finding**

Question asked: How strong does the training bias need to be before our method works or fails?


Method: Build biased training sets with different ratios of aligned vs conflicting samples: 50/50 (no bias) to
99/1 (extreme bias). Measure gap before and after ablation for each ratio.


_Figure 3 — Ablation 3: Bias Ratio Sensitivity. Top: baseline gap (blue) vs ablated gap (orange). Green region = ablation works._

_Red region = ablation fails. Bottom: gap reduction bars and shifted accuracy (red line)._


How to read the Top Panel:


  - Blue line (Baseline Gap): Grows from −0.006 at 50/50 to +0.054 at 99/1. More bias = bigger gap =
unfairer model.

  - Orange line (Ablated Gap): Below baseline at mild bias ratios (50/50 through 90/10) — ABLATION IS
HELPING. Above baseline at 95/5 and 99/1 — ablation fails.

  - Green shaded region (left): Where ablation is effective. Red shaded region (right): Where ablation
fails.


How to read the Bottom Panel:


  - Green bars (Gap Reduction): Positive values (+0.010 to +0.015) mean ablation reduced the gap —
good! Grey bars at 95/5 and 99/1 = no improvement.

  - Red line (Shifted Accuracy after ablation): Falls from 0.989 at 50/50 to 0.938 at 99/1 — extreme bias
degrades performance even after ablation.


#### **Ablation Study 4: Pooling Method Comparison**

Question asked: Does the way we aggregate SAE activations across image patches matter?


Method: Compare Mean Pooling (average across 196 patches) vs Max Pooling (take the max across 196
patches) for building the image-level SAE representation.


_Figure 4 — Ablation 4: Pooling Method Comparison. Purple bars = shortcuts found. Red bars = maximum spurious correlation_

_(r_spur)._


How to read:


  - Mean Pooling: Finds 27 shortcut features. Max spurious correlation = 0.613.

  - Max Pooling: Finds slightly fewer (24) shortcut features, but with stronger individual correlations (r =
0.734).




#### **Ablation Study 5: Seed Stability**

Question asked: Are results consistent across different random seeds, or are they lucky outcomes?


Method: Run the full pipeline 5 times with different random seeds (0, 42, 123, 456, 789). Each seed affects
the random sampling of training data and test sets. Measure shortcuts found and probe accuracy each time.


_Figure 5 — Ablation 5: Seed Stability. Blue bars = shortcuts found per seed. Red diamonds = probe accuracy (perfectly stable at_

_0.890). Blue dashed line = mean. Shaded band = ±1 standard deviation._


How to read:


  - Blue bars (Shortcuts Found): Range from 23 to 31 across 5 seeds. Mean = 26.6, standard deviation =
2.8. This is a tight, stable range.

  - Red line (Probe Accuracy): Exactly 0.890 for every single seed. Perfect stability — the model ALWAYS
encodes dark_border regardless of sampling.


### **6. Mitigation Strategies — How to Fix the Cheating**

After detecting shortcuts, you apply four strategies to reduce or eliminate their effect, and compare them
systematically.

#### **6.1 The Four Strategies Explained**






|Strategy|Graph Label|Graph Color|How It Works|
|---|---|---|---|
|Vanilla (no fix)|Vanilla|Red|Train normally on biased<br>data. Baseline showing<br>the full problem.|
|SAE Ablation Only|Ablation|Blue|At inference: zero out<br>identified shortcut SAE<br>features. Training data<br>stays biased.|
|DRO Reweighting Only|Reweight|Green|During training:<br>oversample conflicting<br>examples until aligned =<br>conflicting count.<br>Training becomes<br>balanced.|
|Combined (Ablation +<br>DRO)|Combined|Purple|Apply BOTH: train with<br>reweighted data AND<br>ablate shortcut features<br>at inference.|


#### **6.2 How Each Strategy Works (With Analogy)**

**Vanilla (Red) — The Problem**


Model trains on 95% aligned + 5% conflicting examples. It overwhelmingly sees the shortcut agreeing with
the label and learns to rely on it. This is the "before" state.


**SAE Ablation (Blue) — Inference-Time Fix**


The biased model is kept as-is. But at test time, a hook intercepts CLIP's internal activations, encodes them
through the SAE, zeros out the 27 identified shortcut features, decodes back, and lets the cleaned
representation continue to the classifier. Training data is still biased — only the representation at inference
is modified.


**DRO Reweighting (Green) — Training-Time Fix**


Before training, the conflicting examples (tumor+low_border, no-tumor+high_border) are repeatedly
resampled (with replacement) until the training set has equal numbers of aligned and conflicting examples.
The classifier then trains on this balanced 50/50 mix. It sees too many counterexamples to rely on the
shortcut.





**Combined (Purple) — Two-Level Fix**


Both strategies applied together: the model is trained on reweighted (balanced) data, AND at inference
time, shortcut features are ablated. This addresses bias at two levels simultaneously: representation level
(ablation) and training level (reweighting).

#### **6.3 The Mitigation Results — Reading the Graphs**


_Figure 6 — Combined Mitigation Results. Left: Shifted Accuracy vs Bias Strength. Middle: Fairness Gap vs Bias Strength. Right:_

_Worst-Group Accuracy at 95/5 bias._


**Left Graph: Shifted Accuracy vs Bias Strength**


Shows how well each strategy performs on the hard (shifted/conflicting) test set as training bias increases
from 50/50 to 99/1.


  - Red (Vanilla): Drops from 0.984 to 0.941 as bias increases. The model gets worse and worse as
training data gets more skewed.


  - Blue (SAE Ablation): Follows Vanilla closely. Ablation alone does not dramatically improve shifted
accuracy.

  - Green (DRO Reweight): Stays flat and HIGH across all bias levels — from 0.984 at 50/50 to 0.988 at
99/1. DRO completely prevents the bias from degrading performance. WINNER for shifted accuracy.

  - Purple (Combined): Nearly identical to DRO — stays high across all ratios.


**Middle Graph: Fairness Gap vs Bias Strength**


Shows the gap (aligned − shifted accuracy) as bias increases. Goal: gap close to 0.


  - Red (Vanilla): Gap grows from ~0 to +0.06 as bias increases. The model becomes increasingly unfair.

  - Blue (SAE Ablation): Slightly reduced gap compared to Vanilla at most ratios, but still grows.

  - Green (DRO Reweight): Gap stays near 0 or slightly negative throughout. This means the model
performs EQUALLY WELL on both aligned and conflicting images.

  - Purple (Combined): Gap stays negative — sometimes even outperforms on the conflicting group.
The combination of both strategies pushes the model beyond just matching — it becomes slightly
better on hard examples.


**Right Graph: Worst-Group Accuracy at 95/5 Bias**


"Worst-group accuracy" measures how well the model does on the HARDEST subgroup of examples — the
key fairness metric. Higher is better.


|Strategy|Worst-Group Acc|Change vs Vanilla|Verdict|
|---|---|---|---|
|Vanilla (Red)|0.947|Baseline|The hardest group gets<br>94.7% accuracy — the<br>biased model|
|Ablation (Blue)|0.942|−0.5% WORSE|Ablation alone slightly<br>hurts worst-group —<br>removes too much signal|
|Reweight (Green)|0.983|+3.6% BEST!|DRO gives the best<br>worst-group accuracy.<br>Most fair model.|
|Combined (Purple)|0.936|−1.1% worse|Combined is slightly<br>below Vanilla for worst-<br>group — ablation and<br>reweighting partially<br>interfere|


### **7. Is This Good Work? — The Scientific Proof**

#### **7.1 Why This Is Valuable Research**


  - Mechanistic Interpretability in Practice: You are not just measuring model outputs — you use SAE to
look INSIDE and identify specific internal features. This is cutting-edge research.

  - Causal Evidence (not just correlation): By ablating specific features and measuring the effect, you
provide causal proof that those features caused the bias. This is much stronger than correlation
alone.

  - Multi-Attribute Scanning: Most shortcut detection research focuses on one spurious attribute. You
scan 8 different attributes simultaneously and rank them.

  - Comparative Mitigation: You test 4 strategies across 6 bias ratios — a systematic, thorough analysis
of when each approach works.

  - Medical AI Safety: Brain tumor diagnosis is life-critical. Demonstrating and fixing bias here has direct
real-world impact.

#### **7.2 Summary of Key Numerical Proofs**

|Proof|Number|What It Shows|
|---|---|---|
|Probe Accuracy|89%|CLIP's features encode the<br>spurious attribute at 89% (vs 50%<br>chance). Bias is inside the model.|
|Shortcut Features Found|27 (dark_border)|27 specific SAE features identified<br>as encoding dark_border more<br>than tumor label.|
|Max r_spurious|0.73|One specific feature correlates<br>with dark_border at r=0.73.<br>Nearly a pure dark_border<br>detector.|
|DRO Worst-Group Gain|+3.6% (0.947 → 0.983)|DRO reweighting improves the<br>hardest group's accuracy by 3.6<br>percentage points.|
|Ablation Gap Reduction|+0.015 at 90/10 bias|Ablation reduces the fairness gap<br>by 1.5 points at moderate bias<br>levels.|
|Seed Stability|Std = 2.8 shortcuts|Results are reproducible across 5<br>different random seeds.|


#### **7.3 Honest Limitations**


  - Ablation alone does not work at extreme bias (95/5, 99/1): The shortcut is too deeply embedded.
DRO is needed.


- Combined strategy has mixed results: It wins on fairness gap but loses on worst-group accuracy vs
DRO-only. This is a real trade-off worth studying further.

- CLIP is frozen: The SAE was trained on frozen CLIP features. A fine-tuned model might have different
(and possibly stronger) shortcut patterns.

- Feature counting is from top-50: You only scan the top 50 most active features. A full scan of all
49,152 features might reveal additional shortcuts.


### **8. Quick Reference Summary**






|Term|Simple Explanation|Your Project Example|
|---|---|---|
|Shortcut|An irrelevant feature used instead<br>of the true signal|Dark border → predict "no tumor"|
|Spurious Correlation|Accidental link between irrelevant<br>feature and label in training data|Hospital A uses dark borders for<br>no-tumor patients|
|Aligned Split|Images where shortcut agrees<br>with truth|Tumor + bright border; No-tumor<br>+ dark border|
|Conflicting Split|Images where shortcut<br>contradicts truth|Tumor + dark border; No-tumor +<br>bright border|
|Shifted Test|Test set where real-world<br>distribution changed|All images from the conflicting<br>pattern|
|Gap|Aligned accuracy minus shifted<br>accuracy|0.05 gap at 99/1 bias for Vanilla<br>model|
|SAE|Sparse Autoencoder — looks<br>inside CLIP and finds individual<br>features|49,152 features, each encoding<br>one concept|
|Ablation|Setting specific SAE features to<br>zero to remove their effect|Zero out 27 dark_border features|
|DRO Reweighting|Oversample minority group<br>during training|Upsample conflicting examples to<br>equal aligned count|
|Probe|Simple classifier trained to detect<br>spurious attribute in CLIP features|Probe accuracy 89% proves CLIP<br>encodes dark_border|
|Worst-Group Acc|Accuracy on the hardest subgroup<br>— key fairness metric|DRO: 0.983 (best); Vanilla: 0.947|




