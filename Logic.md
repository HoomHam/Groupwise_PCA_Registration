# Groupwise Registration Pipeline (16 Timepoints)

## 📦 Input Data

- Source: `.mat` files  
- Loaded as: **NumPy arrays**  
- Shape: `(X, Y, Z, T)` where `T = 16`

### Conversion
- Each 3D volume → `SimpleITK.Image`
- Stacked using `sitk.JoinSeries`

👉 Final working format:
- **4D SimpleITK image** `(X, Y, Z, 16)`

---

## 🛠 Platforms & Methods

- **SimpleITK** → image handling and 4D processing  
- **Elastix / Transformix** → registration and transformation application  
- **ANTs-style groupwise strategy** → conceptual iterative alignment  
- **PCA-based enhancement** → improves contrast and stabilizes registration  

---

## 🧠 Core Strategy

A **hierarchical, progressive groupwise registration pipeline**:

1. Register small stable groups (4 images)
2. Expand to medium groups (7 images)
3. Merge into larger groups (14 images)
4. Align middle phases using averages
5. Apply transformations back to all images
6. Perform final global registration (16 images)

---

## 🔹 Why Groups of 4?

- Motion between adjacent frames is **minimal**
- Registration is more stable and robust
- Natural decomposition:

```
16 → 4 + 4 + 4 + 4
```

---

## 🔸 Phase 1: First 7 Images

### Step 1 — First 4 Images
```
[R1, R2, R3, R4]
```

→ Groupwise registration → compute average:

```
A1 = (R1 + R2 + R3 + R4)'
```

---

### Step 2 — Expand to 7
```
[A1, R5, R6, R7]
```

→ Register again

---

### Step 3 — Transformation Propagation

For each Ri:
- Move Ri to first position
- Fill remaining positions with 3 other images
- Apply transformations

Result:
```
[TR1, TR2, TR3, TR4]
```

---

### Step 4 — Build First 7 Set
```
[TR1, TR2, TR3, TR4, R5', R6', R7']
```

→ Register:
```
(R1 + ... + R7)"
```

---

## 🔸 Phase 2: Last 7 Images (Symmetric)

Original:
```
[R10, R11, R12, R13, R14, R15, R16]
```

Reverse:
```
[R16, R15, R14, R13, R12, R11, R10]
```

---

### Step 1 — First 4 from End
```
[R16, R15, R14, R13]
```

→ Register → compute average:

```
A2 = (R16 + ... + R13)'
```

---

### Step 2 — Expand
```
[A2, R12, R11, R10]
```

→ Register

---

### Step 3 — Transformation Propagation
Same logic as first group

---

### Step 4 — Final Last 7
```
(R16 + ... + R10)"
```

→ Reverse back to original order

---

## 🔸 Phase 3: Combine 14 Images

```
[First 7] + [Last 7]
```

→ Register:
```
14-image group
```

---

## 🔸 Phase 4: Midpoint Alignment

Compute averages:
```
A_first = mean(R1–R7)
A_last  = mean(R10–R16)
```

---

### Build midpoint set:
```
[A_first, R8, R9, A_last]
```

→ Register

👉 This step bridges early, middle, and late phases

---

## 🔸 Phase 5: Apply Transformations Back

### First 7 (BEGINNING strategy)
For each Ri:
```
[Ri, arbitrary, arbitrary, arbitrary]
```

→ Extract transformed Ri

---

### Last 7 (END strategy)
For each Ri:
```
[arbitrary, arbitrary, arbitrary, Ri]
```

→ Extract transformed Ri

---

## 🔸 Phase 6: Final Assembly

```
[TR1–TR7] + [R8', R9'] + [TR10–TR16]
```

---

## 🔸 Phase 7: Final Global Registration

→ Register all 16 images:

```
Final aligned 4D dataset
```

---

## 🧠 Why This Works

- **Hierarchical registration** → avoids instability  
- **Averaging** → reduces noise and creates anchors  
- **Transformation reuse** → preserves consistency  
- **Symmetry (start/end)** → unbiased alignment  
- **PCA enhancement** → improves convergence  

---

## 🔥 One-line Summary

A hierarchical, PCA-enhanced groupwise registration pipeline that aligns 16 timepoints by progressively expanding groups (4→7→14→16) and propagating transformations using Elastix within SimpleITK.