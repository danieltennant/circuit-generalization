# Linear Algebra for Mechanistic Interpretability
### Pencil-and-Paper Exercises

**Who this is for:** Someone with a CS background whose linear algebra is rusty, working toward understanding how transformer attention works mechanically.

**How to use this:** Work through each section in order. The exercises build on each other. Do them by hand — the point is to build intuition, not to get answers fast. Check your work against the solutions at the end.

---

## Section 1: Vectors and Dot Products

**Why this matters for transformers:** Every token in a transformer is represented as a vector (its "embedding"). When the model computes attention, it is fundamentally asking: *how similar is this query vector to each key vector?* That similarity is measured with a dot product. If you can read a dot product and know what it means geometrically, you can read an attention pattern and know what the model is "looking at."

---

**Exercise 1.1** — Basic dot product (2D)

Compute the dot product of:

$$\mathbf{a} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$$

Recall: $\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2$

---

**Exercise 1.2** — Basic dot product (3D)

Compute:

$$\mathbf{u} = \begin{bmatrix} 1 \\ -2 \\ 3 \end{bmatrix} \cdot \begin{bmatrix} 4 \\ 1 \\ 2 \end{bmatrix}$$

---

**Exercise 1.3** — Geometric interpretation

You are told that $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$, where $\theta$ is the angle between the vectors.

For each case below, say whether the dot product will be **positive**, **negative**, or **zero** — and explain in one sentence what that means geometrically:

(a) Two vectors pointing in exactly the same direction.
(b) Two vectors pointing in exactly opposite directions.
(c) Two vectors that are perpendicular (orthogonal).
(d) A long vector and a short vector pointing in the same direction, compared to two unit vectors pointing in the same direction. Which dot product is larger?

---

**Exercise 1.4** — Dot product as similarity

Consider three 2D word embedding vectors (toy, made-up numbers):

$$\text{"cat"} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}, \quad \text{"dog"} = \begin{bmatrix} 2 \\ 2 \end{bmatrix}, \quad \text{"car"} = \begin{bmatrix} -3 \\ 1 \end{bmatrix}$$

(a) Compute $\text{"cat"} \cdot \text{"dog"}$
(b) Compute $\text{"cat"} \cdot \text{"car"}$
(c) Which pair is more "similar" according to the dot product? Does this match your intuition about the words?

---

**Exercise 1.5** — Normalisation and dot products

Two vectors:

$$\mathbf{p} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}, \quad \mathbf{q} = \begin{bmatrix} 0 \\ 5 \end{bmatrix}$$

(a) Compute $\|\mathbf{p}\| = \sqrt{p_1^2 + p_2^2}$ and $\|\mathbf{q}\|$.
(b) Compute $\mathbf{p} \cdot \mathbf{q}$.
(c) Now normalise both vectors (divide each by its magnitude) to get $\hat{\mathbf{p}}$ and $\hat{\mathbf{q}}$.
(d) Compute $\hat{\mathbf{p}} \cdot \hat{\mathbf{q}}$. This is the cosine similarity. Why might cosine similarity be a more useful measure of directional similarity than the raw dot product?

---

## Section 2: Matrix Multiplication

**Why this matters for transformers:** The weight matrices $W_Q$, $W_K$, $W_V$, and $W_O$ are all matrices. When a token embedding passes through one of these, it is a matrix multiplication. Understanding what matrix multiplication does — and critically, what *shape* the result has — is essential for reading transformer code without getting lost.

---

**Exercise 2.1** — 2x2 matrix multiplication

Compute $AB$ where:

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

Recall: the entry in row $i$, column $j$ of $AB$ is the dot product of row $i$ of $A$ with column $j$ of $B$.

---

**Exercise 2.2** — Non-square matrix multiplication

Compute $AB$ where:

$$A = \begin{bmatrix} 1 & 0 & 2 \\ 3 & 1 & 1 \end{bmatrix} \quad (2 \times 3), \quad B = \begin{bmatrix} 1 & 2 \\ 0 & 1 \\ 4 & 0 \end{bmatrix} \quad (3 \times 2)$$

(a) What shape is $AB$?
(b) Compute $AB$.
(c) Can you compute $BA$? If yes, what shape is it? If no, why not?

---

**Exercise 2.3** — Shape tracking

For each of the following, state whether the multiplication is valid, and if so, the shape of the result. You do **not** need to compute the values.

(a) $A$ is $(3 \times 4)$, $B$ is $(4 \times 2)$. Compute $AB$.
(b) $A$ is $(5 \times 3)$, $B$ is $(5 \times 3)$. Compute $AB$.
(c) $A$ is $(1 \times 6)$ (a row vector), $B$ is $(6 \times 1)$ (a column vector). Compute $AB$.
(d) $A$ is $(6 \times 1)$, $B$ is $(1 \times 6)$. Compute $AB$. How is this different from (c)?
(e) In a transformer, a token embedding $\mathbf{x}$ has shape $(d_{\text{model}},)$, which we treat as $(1 \times d_{\text{model}})$. The query weight matrix $W_Q$ has shape $(d_{\text{model}} \times d_{\text{head}})$. What shape is $\mathbf{x} W_Q$?

---

**Exercise 2.4** — $W_V$ and $W_O$ in miniature

In a transformer attention head, the value matrix $W_V$ maps each token to a "value" vector, and $W_O$ maps the aggregated value back to the residual stream. Here is a tiny version.

Let:

$$W_V = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}, \quad W_O = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$

And let token embedding $\mathbf{x} = \begin{bmatrix} 3 & 1 \end{bmatrix}$ (a row vector).

(a) Compute $\mathbf{v} = \mathbf{x} W_V$ (the value vector for this token).
(b) Compute the output $\mathbf{o} = \mathbf{v} W_O$.
(c) What is the combined effect $\mathbf{x} (W_V W_O)$? Compute $W_V W_O$ first, then multiply by $\mathbf{x}$.
(d) Does your answer to (c) match your answer to (b)? It should — this is matrix associativity: $(AB)C = A(BC)$.

---

**Exercise 2.5** — The OV circuit

This exercise previews one of the key ideas in mechanistic interpretability: the **OV circuit**, which describes what a head *does* to the residual stream regardless of what it attends to.

The OV matrix is defined as $W_{OV} = W_V W_O$.

Using:
$$W_V = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}, \quad W_O = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

(a) Compute $W_{OV} = W_V W_O$.
(b) What does $W_O = I$ (the identity matrix) imply about what $W_O$ is doing here?
(c) Apply $W_{OV}$ to two different token embeddings: $\mathbf{x}_1 = \begin{bmatrix} 1 & 0 \end{bmatrix}$ and $\mathbf{x}_2 = \begin{bmatrix} 0 & 1 \end{bmatrix}$. What do you notice about the relationship between the input and output?

---

## Section 3: Linear Transformations

**Why this matters for transformers:** A matrix is not just an array of numbers — it is a *function*. When you multiply a vector by a matrix, you are applying a linear transformation: stretching, rotating, or projecting that vector in space. Every weight matrix in a transformer is doing this to token representations. If you can think of matrices as functions, you can think about what information they extract or suppress.

---

**Exercise 3.1** — Matrix as function

Let $T = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$.

Apply $T$ to each of the following vectors and describe what geometric effect $T$ has:

(a) $\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
(b) $\mathbf{v} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$
(c) $\mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$
(d) In one sentence, what does $T$ do to any 2D vector?

---

**Exercise 3.2** — Two matrices, same input

Let:
$$A = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}, \quad B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad \mathbf{v} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}$$

(a) Compute $A\mathbf{v}$. What does $A$ do geometrically?
(b) Compute $B\mathbf{v}$. What does $B$ do geometrically?
(c) These are two different "lenses" through which to view the same vector $\mathbf{v}$. In a transformer, different attention heads apply different $W_Q, W_K, W_V$ matrices to the same token — they are asking different questions about the same information. Does this analogy feel concrete now?

---

**Exercise 3.3** — Matrix multiplication as function composition

Let $f(\mathbf{v}) = A\mathbf{v}$ and $g(\mathbf{v}) = B\mathbf{v}$ where:

$$A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad B = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}, \quad \mathbf{v} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$

(a) Compute $f(\mathbf{v}) = A\mathbf{v}$.
(b) Compute $g(f(\mathbf{v})) = B(A\mathbf{v})$ by applying $B$ to your answer from (a).
(c) Now compute $C = BA$ directly.
(d) Compute $C\mathbf{v}$.
(e) Do (b) and (d) agree? They should — matrix multiplication *is* function composition: applying $A$ then $B$ is identical to applying the single matrix $BA$.

---

**Exercise 3.4** — Projections

A projection matrix onto the x-axis is:

$$P = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$$

(a) Apply $P$ to $\mathbf{v} = \begin{bmatrix} 4 \\ 7 \end{bmatrix}$. What information is lost?
(b) Apply $P$ twice: compute $P(P\mathbf{v})$. What do you notice?
(c) A matrix $M$ satisfying $M^2 = M$ is called **idempotent**. Verify that $P^2 = P$. Why does this make sense geometrically? (Projecting something that's already projected should do nothing.)
(d) In transformers, the residual stream idea means that information is *added to* (not replaced by) the stream. How is this different from applying a projection?

---

## Section 4: Attention Scores by Hand

**Why this matters:** This is the core computation. In a transformer attention head, each token generates a **query** (what am I looking for?) and a **key** (what do I offer?). The attention score between token $i$ and token $j$ is the dot product of token $i$'s query with token $j$'s key, scaled and softmaxed. Working through this by hand — with real numbers — makes the formula completely concrete.

The full formula for attention is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_{\text{head}}}}\right) V$$

In this section you will compute the $QK^T / \sqrt{d_{\text{head}}}$ part and the softmax. We use $d_{\text{head}} = 2$ throughout.

---

**Exercise 4.1** — Computing Q and K vectors

Suppose we have a 3-token sequence. The token embeddings (already 2D for simplicity) are:

$$\mathbf{x}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{x}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad \mathbf{x}_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

The query and key weight matrices are:

$$W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W_K = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$$

(a) Compute the query vectors $\mathbf{q}_i = \mathbf{x}_i^T W_Q$ for $i = 1, 2, 3$.
(b) Compute the key vectors $\mathbf{k}_i = \mathbf{x}_i^T W_K$ for $i = 1, 2, 3$.

*Note:* We write embeddings as row vectors here and multiply $\mathbf{x}^T W$ rather than $W^T \mathbf{x}$ — this matches the convention used in ARENA and most transformer implementations.

---

**Exercise 4.2** — Raw attention scores

Using your Q and K vectors from Exercise 4.1:

(a) Compute the raw attention score $s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j$ for every pair $(i, j)$ where $i, j \in \{1, 2, 3\}$. Fill in this 3x3 table:

|         | $\mathbf{k}_1$ | $\mathbf{k}_2$ | $\mathbf{k}_3$ |
|---------|:--------------:|:--------------:|:--------------:|
| $\mathbf{q}_1$ | $s_{11}$ | $s_{12}$ | $s_{13}$ |
| $\mathbf{q}_2$ | $s_{21}$ | $s_{22}$ | $s_{23}$ |
| $\mathbf{q}_3$ | $s_{31}$ | $s_{32}$ | $s_{33}$ |

(b) Write the full attention score matrix $S$ (3x3).
(c) Note that row $i$ of $S$ contains the scores for token $i$ attending to every other token. Which token does token 3 most "want to attend to" based on raw scores?

---

**Exercise 4.3** — Scaling and softmax

**Step 1: Scale**

Divide every entry in your score matrix $S$ by $\sqrt{d_{\text{head}}} = \sqrt{2} \approx 1.414$.

Call the result $\tilde{S}$. (Round to 2 decimal places.)

**Step 2: Softmax**

The softmax of a row vector $\mathbf{z} = [z_1, z_2, \ldots, z_n]$ is:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Apply softmax to each row of $\tilde{S}$ independently. This gives the **attention pattern** $A$ — a 3x3 matrix where each row sums to 1.

Useful values: $e^0 \approx 1.000$, $e^{0.71} \approx 2.034$, $e^{1.41} \approx 4.096$, $e^{2.12} \approx 8.331$.

(a) Compute the scaled scores for row 1 (token 1's query against all keys).
(b) Apply softmax to row 1. Verify it sums to 1.
(c) Apply softmax to row 3 (you can leave row 2 as an exercise if time is short).
(d) Interpret the attention pattern: what is token 1 "attending to"? What is token 3 "attending to"?

---

## Section 5: Tensor Shapes and Einsum Notation

**Why this matters:** Modern transformer code uses `einops.einsum` and `torch.einsum` extensively because they make the intention of each operation explicit via index notation. Being able to read an einsum string and immediately know the input/output shapes — and what computation is happening — is a superpower when reading or debugging transformer code. The ARENA curriculum uses these constantly.

**How to read einsum notation:** Each letter is a dimension. A letter appearing in both inputs is summed over (contracted). A letter appearing in the output is kept. Shapes must be consistent: the same letter must have the same size everywhere it appears.

---

**Exercise 5.1** — Basic einsum interpretation

For each einsum expression, state:
- What each letter represents (you can name them anything sensible)
- The shapes of the inputs
- The shape of the output
- In plain English, what operation is being performed

**(a)** `"i, i -> "` applied to tensors `A` and `B` where `A` has shape `(4,)` and `B` has shape `(4,)`.

**(b)** `"i j, j k -> i k"` where `A` has shape `(3, 4)` and `B` has shape `(4, 5)`.

**(c)** `"b i j, b j k -> b i k"` where `A` has shape `(8, 3, 4)` and `B` has shape `(8, 4, 5)`.
(This is batched matrix multiplication — `b` is the batch dimension.)

**(d)** `"i j -> j i"` where `A` has shape `(3, 7)`.

---

**Exercise 5.2** — Shape consistency check

For each of the following, identify the error (if any) in the einsum expression given the tensor shapes:

(a) `"i j, i k -> j k"` with `A` shape `(3, 4)` and `B` shape `(3, 5)`.
(b) `"i j, j k -> i k"` with `A` shape `(3, 4)` and `B` shape `(5, 2)`.
(c) `"i j k, i j -> i k"` with `A` shape `(2, 3, 4)` and `B` shape `(2, 3)`.

---

**Exercise 5.3** — The ARENA 1.3 einsum

The following einsum appears in ARENA exercise 1.3 for computing attention scores:

```python
einops.einsum(Q, K, "seqQ n h, seqK n h -> n seqQ seqK")
```

Where the variables have these meanings:
- `seqQ`: number of query positions (sequence length)
- `seqK`: number of key positions (sequence length — same value as `seqQ` in self-attention)
- `n`: number of attention heads
- `h`: head dimension ($d_{\text{head}}$)

(a) What are the shapes of `Q` and `K`?
(b) What is the shape of the output?
(c) Which dimension is being summed over (contracted)? What does summing over `h` correspond to mathematically?
(d) The output has shape `(n, seqQ, seqK)`. What does entry `[n, i, j]` of this tensor represent?
(e) Why is `n` in the output rather than being contracted? What would it mean if we had written `"seqQ n h, seqK n h -> seqQ seqK"` instead?

---

**Exercise 5.4** — Writing your own einsum

Write the einsum string for each of the following operations:

(a) Standard (non-batched) matrix multiplication of `A` (shape `m x n`) with `B` (shape `n x p`) to get output of shape `m x p`.

(b) For each token in a sequence, compute the dot product of that token's vector with a fixed vector `w`. Input: `X` of shape `(seq, d)`, `w` of shape `(d,)`. Output shape: `(seq,)`.

(c) Compute the outer product of two vectors `u` of shape `(m,)` and `v` of shape `(n,)` to get a matrix of shape `(m, n)`.

(d) **Challenge:** You have value vectors `V` of shape `(seqK, n, h)` and an attention pattern `A` of shape `(n, seqQ, seqK)`. Write the einsum that computes the weighted sum of values for each query position and head. Output shape: `(seqQ, n, h)`.

---
---

# Worked Solutions

---

## Solutions: Section 1

**1.1**

$$\mathbf{a} \cdot \mathbf{b} = (3)(2) + (1)(4) = 6 + 4 = \mathbf{10}$$

---

**1.2**

$$\mathbf{u} \cdot \mathbf{v} = (1)(4) + (-2)(1) + (3)(2) = 4 - 2 + 6 = \mathbf{8}$$

---

**1.3**

(a) $\theta = 0°$, so $\cos\theta = 1$. Dot product is **positive**. Vectors pointing the same way are maximally aligned.
(b) $\theta = 180°$, so $\cos\theta = -1$. Dot product is **negative**. The vectors point in completely opposite directions.
(c) $\theta = 90°$, so $\cos\theta = 0$. Dot product is **zero**. Orthogonal vectors share no directional component.
(d) The dot product of the long and short vector is **larger**, because $\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ scales with magnitude. The raw dot product conflates direction and magnitude — which is why cosine similarity (see 1.5) is often preferred.

---

**1.4**

(a) $\text{"cat"} \cdot \text{"dog"} = (2)(2) + (3)(2) = 4 + 6 = \mathbf{10}$
(b) $\text{"cat"} \cdot \text{"car"} = (2)(-3) + (3)(1) = -6 + 3 = \mathbf{-3}$
(c) "cat" and "dog" are more similar by dot product (10 vs -3). This matches intuition — both are animals/pets, while "car" is unrelated.

---

**1.5**

(a) $\|\mathbf{p}\| = \sqrt{9 + 16} = \sqrt{25} = 5$. $\|\mathbf{q}\| = \sqrt{0 + 25} = 5$.
(b) $\mathbf{p} \cdot \mathbf{q} = (3)(0) + (4)(5) = 20$.
(c) $\hat{\mathbf{p}} = \begin{bmatrix} 3/5 \\ 4/5 \end{bmatrix}$, $\hat{\mathbf{q}} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$.
(d) $\hat{\mathbf{p}} \cdot \hat{\mathbf{q}} = (3/5)(0) + (4/5)(1) = 4/5 = 0.8$.

Cosine similarity removes the effect of vector length, measuring only the *angle* between directions. Two vectors can have a large raw dot product simply because they are long, even if they point in very different directions. Cosine similarity gives a value in $[-1, 1]$ that purely reflects directional alignment.

---

## Solutions: Section 2

**2.1**

$$AB = \begin{bmatrix} (1)(5)+(2)(7) & (1)(6)+(2)(8) \\ (3)(5)+(4)(7) & (3)(6)+(4)(8) \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

---

**2.2**

(a) $A$ is $2 \times 3$, $B$ is $3 \times 2$. Inner dimensions match (both 3), so $AB$ has shape $\mathbf{2 \times 2}$.

(b)
$$AB = \begin{bmatrix} (1)(1)+(0)(0)+(2)(4) & (1)(2)+(0)(1)+(2)(0) \\ (3)(1)+(1)(0)+(1)(4) & (3)(2)+(1)(1)+(1)(0) \end{bmatrix} = \begin{bmatrix} 9 & 2 \\ 7 & 7 \end{bmatrix}$$

(c) Yes, $BA$ is valid: $B$ is $3 \times 2$, $A$ is $2 \times 3$, so $BA$ has shape $\mathbf{3 \times 3}$.

$$BA = \begin{bmatrix} (1)(1)+(2)(3) & (1)(0)+(2)(1) & (1)(2)+(2)(1) \\ (0)(1)+(1)(3) & (0)(0)+(1)(1) & (0)(2)+(1)(1) \\ (4)(1)+(0)(3) & (4)(0)+(0)(1) & (4)(2)+(0)(1) \end{bmatrix} = \begin{bmatrix} 7 & 2 & 4 \\ 3 & 1 & 1 \\ 4 & 0 & 8 \end{bmatrix}$$

Note: $AB \neq BA$ in general, and they don't even have the same shape here.

---

**2.3**

(a) Valid. Inner dimensions: 4 matches 4. Output shape: $\mathbf{(3 \times 2)}$.
(b) **Not valid.** Inner dimensions: 3 (from A) does not match 5 (from B).
(c) Valid. $(1 \times 6)(6 \times 1)$. Output shape: $\mathbf{(1 \times 1)}$ — a scalar. This is a dot product.
(d) Valid. $(6 \times 1)(1 \times 6)$. Output shape: $\mathbf{(6 \times 6)}$ — an outer product, a full matrix.
(e) $(1 \times d_{\text{model}})(d_{\text{model}} \times d_{\text{head}}) = \mathbf{(1 \times d_{\text{head}})}$. The query vector for this token has $d_{\text{head}}$ dimensions.

---

**2.4**

(a) $\mathbf{v} = \mathbf{x} W_V = \begin{bmatrix} 3 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 3 & 2 \end{bmatrix}$

(b) $\mathbf{o} = \mathbf{v} W_O = \begin{bmatrix} 3 & 2 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 2 & 3 \end{bmatrix}$

(c) First, $W_V W_O = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 2 & 0 \end{bmatrix}$

Then $\mathbf{x}(W_V W_O) = \begin{bmatrix} 3 & 1 \end{bmatrix}\begin{bmatrix} 0 & 1 \\ 2 & 0 \end{bmatrix} = \begin{bmatrix} 2 & 3 \end{bmatrix}$

(d) Yes — both give $\begin{bmatrix} 2 & 3 \end{bmatrix}$. Associativity holds: $(\mathbf{x} W_V) W_O = \mathbf{x} (W_V W_O)$.

---

**2.5**

(a) $W_{OV} = W_V W_O = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}$

(b) $W_O = I$ means $W_O$ does nothing — it is the identity transformation. In this case $W_{OV} = W_V$ exactly. In practice $W_O \neq I$, and the composition $W_V W_O$ determines what information each head writes back to the residual stream.

(c) $\mathbf{x}_1 W_{OV} = \begin{bmatrix} 1 & 0 \end{bmatrix}\begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} = \begin{bmatrix} 2 & 1 \end{bmatrix}$

$\mathbf{x}_2 W_{OV} = \begin{bmatrix} 0 & 1 \end{bmatrix}\begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} = \begin{bmatrix} 0 & 3 \end{bmatrix}$

The OV matrix consistently transforms the input: $\mathbf{x}_1$ (the "first basis direction") gets mapped to $[2, 1]$, and $\mathbf{x}_2$ gets mapped to $[0, 3]$. The OV circuit tells you *what the head writes* into the residual stream when it attends to a given token — independent of *where* it attends.

---

## Solutions: Section 3

**3.1**

(a) $T\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}2\\0\end{bmatrix}$ — stretched by 2 in the x-direction.
(b) $T\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}0\\3\end{bmatrix}$ — stretched by 3 in the y-direction.
(c) $T\begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix}2\\3\end{bmatrix}$ — stretched by 2 in x and 3 in y simultaneously.
(d) $T$ is a **scaling transformation** — it stretches the x-axis by factor 2 and the y-axis by factor 3, independently.

---

**3.2**

(a) $A\mathbf{v} = \begin{bmatrix}1 & 0\\0 & -1\end{bmatrix}\begin{bmatrix}3\\2\end{bmatrix} = \begin{bmatrix}3\\-2\end{bmatrix}$. $A$ **reflects** across the x-axis (negates the y-component).
(b) $B\mathbf{v} = \begin{bmatrix}0 & 1\\1 & 0\end{bmatrix}\begin{bmatrix}3\\2\end{bmatrix} = \begin{bmatrix}2\\3\end{bmatrix}$. $B$ **swaps** the x and y components — a reflection across the diagonal.
(c) This is the key intuition: different weight matrices ask different questions about the same token vector. One head might be looking for "is this a noun?" while another looks for "is this at the start of a phrase?" — all operating on the same residual stream vector.

---

**3.3**

(a) $A\mathbf{v} = \begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = \begin{bmatrix}3\\2\end{bmatrix}$

(b) $B(A\mathbf{v}) = \begin{bmatrix}2 & 0\\0 & 2\end{bmatrix}\begin{bmatrix}3\\2\end{bmatrix} = \begin{bmatrix}6\\4\end{bmatrix}$

(c) $C = BA = \begin{bmatrix}2 & 0\\0 & 2\end{bmatrix}\begin{bmatrix}1 & 1\\0 & 1\end{bmatrix} = \begin{bmatrix}2 & 2\\0 & 2\end{bmatrix}$

(d) $C\mathbf{v} = \begin{bmatrix}2 & 2\\0 & 2\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = \begin{bmatrix}6\\4\end{bmatrix}$

(e) Yes, both give $\begin{bmatrix}6\\4\end{bmatrix}$. Applying $A$ then $B$ is the same as applying the single matrix $C = BA$. This is why the OV circuit ($W_V W_O$) can be analysed as a single matrix — it is the composition of two linear functions.

---

**3.4**

(a) $P\mathbf{v} = \begin{bmatrix}1 & 0\\0 & 0\end{bmatrix}\begin{bmatrix}4\\7\end{bmatrix} = \begin{bmatrix}4\\0\end{bmatrix}$. The y-component (7) is lost — it cannot be recovered.

(b) $P(P\mathbf{v}) = P\begin{bmatrix}4\\0\end{bmatrix} = \begin{bmatrix}4\\0\end{bmatrix}$. Projecting again does nothing — the result is unchanged.

(c) $P^2 = PP = \begin{bmatrix}1&0\\0&0\end{bmatrix}\begin{bmatrix}1&0\\0&0\end{bmatrix} = \begin{bmatrix}1&0\\0&0\end{bmatrix} = P$. Geometrically: once a vector is on the x-axis, projecting it onto the x-axis again leaves it exactly where it is.

(d) The residual stream uses **addition** ($\mathbf{x}_{\text{new}} = \mathbf{x} + \Delta\mathbf{x}$), not replacement. Nothing is discarded — each layer adds information to the stream. A projection destroys information; the residual connection preserves it.

---

## Solutions: Section 4

**4.1**

Since $W_Q = I$ (the identity), query vectors equal the embeddings:

$$\mathbf{q}_1 = \begin{bmatrix}1 & 0\end{bmatrix}, \quad \mathbf{q}_2 = \begin{bmatrix}0 & 1\end{bmatrix}, \quad \mathbf{q}_3 = \begin{bmatrix}1 & 1\end{bmatrix}$$

For keys, $W_K = \begin{bmatrix}2&0\\0&1\end{bmatrix}$ doubles the first component:

$$\mathbf{k}_1 = \begin{bmatrix}1&0\end{bmatrix}\begin{bmatrix}2&0\\0&1\end{bmatrix} = \begin{bmatrix}2&0\end{bmatrix}$$
$$\mathbf{k}_2 = \begin{bmatrix}0&1\end{bmatrix}\begin{bmatrix}2&0\\0&1\end{bmatrix} = \begin{bmatrix}0&1\end{bmatrix}$$
$$\mathbf{k}_3 = \begin{bmatrix}1&1\end{bmatrix}\begin{bmatrix}2&0\\0&1\end{bmatrix} = \begin{bmatrix}2&1\end{bmatrix}$$

---

**4.2**

Computing $s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j$:

- $s_{11} = (1)(2)+(0)(0) = 2$
- $s_{12} = (1)(0)+(0)(1) = 0$
- $s_{13} = (1)(2)+(0)(1) = 2$
- $s_{21} = (0)(2)+(1)(0) = 0$
- $s_{22} = (0)(0)+(1)(1) = 1$
- $s_{23} = (0)(2)+(1)(1) = 1$
- $s_{31} = (1)(2)+(1)(0) = 2$
- $s_{32} = (1)(0)+(1)(1) = 1$
- $s_{33} = (1)(2)+(1)(1) = 3$

$$S = \begin{bmatrix} 2 & 0 & 2 \\ 0 & 1 & 1 \\ 2 & 1 & 3 \end{bmatrix}$$

(c) Row 3 is $[2, 1, 3]$. The highest score is $s_{33} = 3$, so token 3 most strongly attends to **token 3** (itself).

---

**4.3**

**Step 1: Scale** by $1/\sqrt{2} \approx 0.707$:

$$\tilde{S} = \begin{bmatrix} 1.41 & 0 & 1.41 \\ 0 & 0.71 & 0.71 \\ 1.41 & 0.71 & 2.12 \end{bmatrix}$$

**Step 2: Softmax**

(a) Row 1 scaled scores: $[1.41,\ 0,\ 1.41]$

(b) Exponentiate: $e^{1.41} \approx 4.096$, $e^0 = 1.000$, $e^{1.41} \approx 4.096$

Sum $= 4.096 + 1.000 + 4.096 = 9.192$

$$A_{1,\cdot} = \left[\frac{4.096}{9.192},\ \frac{1.000}{9.192},\ \frac{4.096}{9.192}\right] \approx [0.446,\ 0.109,\ 0.446]$$

Sum check: $0.446 + 0.109 + 0.446 \approx 1.0$. Correct.

(c) Row 3 scaled scores: $[1.41,\ 0.71,\ 2.12]$

Exponentiate: $e^{1.41} \approx 4.096$, $e^{0.71} \approx 2.034$, $e^{2.12} \approx 8.331$

Sum $= 4.096 + 2.034 + 8.331 = 14.461$

$$A_{3,\cdot} = \left[\frac{4.096}{14.461},\ \frac{2.034}{14.461},\ \frac{8.331}{14.461}\right] \approx [0.283,\ 0.141,\ 0.576]$$

(d) **Interpretation:**

- Token 1 splits its attention equally between itself and token 3 (both ~44.6%), largely ignoring token 2 (~10.9%). Token 1's query has only a first-component, and tokens 1 and 3 both have large first components in their keys (due to $W_K$ doubling the first dimension).
- Token 3 attends most strongly to itself (~57.6%), moderately to token 1 (~28.3%), and least to token 2 (~14.1%). Token 3's query has components in both dimensions, but the large key of token 3 (boosted by $W_K$) makes it the most attractive target.

---

## Solutions: Section 5

**5.1**

(a) `"i, i -> "` — Both inputs have shape `(4,)`. Output has shape `()` — a scalar. **This is a dot product.** The index `i` is contracted (summed over), leaving nothing.

(b) `"i j, j k -> i k"` — `A` is `(3, 4)`, `B` is `(4, 5)`. Contracting over `j = 4`. Output is `(3, 5)`. **Standard matrix multiplication.**

(c) `"b i j, b j k -> b i k"` — `A` is `(8, 3, 4)`, `B` is `(8, 4, 5)`. Contracting over `j = 4`, keeping batch `b = 8`. Output is `(8, 3, 5)`. **Batched matrix multiplication** — 8 independent matrix multiplications run simultaneously.

(d) `"i j -> j i"` — `A` is `(3, 7)`. No contraction. Output is `(7, 3)`. **Transpose.**

---

**5.2**

(a) `"i j, i k -> j k"` with shapes `(3, 4)` and `(3, 5)`. **Valid — no error.** `i = 3` is contracted, `j = 4` and `k = 5` are kept. Output shape: `(4, 5)`. This computes $A^T B$.

(b) `"i j, j k -> i k"` with shapes `(3, 4)` and `(5, 2)`. **Error.** Index `j` must be the same size in both inputs: it is 4 in `A` but 5 in `B`. The inner dimensions don't align — this is the einsum equivalent of a shape mismatch in matrix multiplication.

(c) `"i j k, i j -> i k"` with shapes `(2, 3, 4)` and `(2, 3)`. **Valid — no error.** `j = 3` is contracted. `i = 2` and `k = 4` are kept. Output shape: `(2, 4)`.

---

**5.3**

(a) `Q` has shape `(seqQ, n, h)` and `K` has shape `(seqK, n, h)`.

(b) Output shape: `(n, seqQ, seqK)`.

(c) `h` is contracted (it appears in both inputs but not the output). Mathematically, summing over `h` computes the **dot product** in the head dimension. For a fixed head `n`, query position `i`, and key position `j`, the output is $\sum_{h} Q[i, n, h] \times K[j, n, h]$ — exactly $\mathbf{q}_{i,n} \cdot \mathbf{k}_{j,n}$.

(d) Entry `[n, i, j]` is the **raw attention score** for head `n`: the dot product of query at position `i` with key at position `j`. After dividing by $\sqrt{d_{\text{head}}}$ and applying softmax over the `seqK` dimension, this becomes the attention weight.

(e) `n` is in the output because each head computes its own independent attention pattern — we want to keep them separate, not sum across heads. If we wrote `"seqQ n h, seqK n h -> seqQ seqK"`, we would sum over `n` (and `h`), collapsing all heads into a single attention score matrix. This would destroy the multi-head structure entirely and is not how transformers work.

---

**5.4**

(a) Standard matrix multiplication:
```
"m n, n p -> m p"
```

(b) Dot product of each token with a fixed vector:
```
"s d, d -> s"
```

(c) Outer product of two vectors:
```
"m, n -> m n"
```
No indices are contracted — `m` and `n` both appear only in one input and in the output.

(d) Weighted sum of values by attention pattern:
```
"seqK n h, n seqQ seqK -> seqQ n h"
```
`seqK` is contracted (the dimension being summed over in the weighted average). `n`, `seqQ`, and `h` are kept. For each query position and head, this computes the weighted combination of value vectors — the aggregation step that produces the attention head output.

---

*End of exercises and solutions.*
