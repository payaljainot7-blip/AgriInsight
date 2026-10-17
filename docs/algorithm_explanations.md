# Comprehensive Algorithm Explanations & Complexity Analysis

This document provides a thorough theoretical and practical breakdown of the four frequent itemset mining algorithms implemented in AgriInsight.

---

## 1. Apriori Algorithm (Agrawal & Srikant, 1994)

### Operating Principle
Apriori uses a **breadth-first (level-wise)** candidate generation and test approach. It is built upon the fundamental **Apriori Principle (Anti-monotone property of support)**:
> If an itemset is frequent, all of its non-empty subsets must also be frequent. Conversely, if any subset is infrequent, the superset cannot be frequent.

### Algorithmic Steps
1. **$L_1$ Generation**: Scan the database once to count support for each individual 1-item. Discard items with count $< \text{min\_count}$.
2. **Candidate Generation ($C_k$ from $L_{k-1}$)**:
   - **Join Step**: Two frequent $(k-1)$-itemsets $P$ and $Q$ are joined if their first $k-2$ items are identical:
     $$P[1:k-2] == Q[1:k-2] \quad \text{and} \quad P[k-1] < Q[k-1]$$
   - **Prune Step**: For each candidate in $C_k$, generate all its $(k-1)$-subsets. If any subset $\notin L_{k-1}$, prune the candidate immediately without scanning transactions.
3. **Candidate Counting**: Scan transactions to compute the frequency of candidates in $C_k$.
4. **Iterate**: Repeat until no candidates qualify as frequent.

### Complexity
- **Time**: $\mathcal{O}(2^{|I|} \times N)$, where $|I|$ is number of frequent items and $N$ is number of transactions. Repeated database scans at each level $k$ cause significant I/O overhead.
- **Space**: $\mathcal{O}(|C_k|)$ to store level candidates. At intermediate levels (e.g. $k=2, 3$), candidate sets can become very large.

---

## 2. FP-Growth Algorithm (Han, Pei, & Yin, 2000)

### Operating Principle
FP-Growth (**Frequent Pattern Growth**) eliminates costly candidate generation by adopting a **divide-and-conquer** pattern-growth paradigm:
1. Compresses the transaction database into a compact memory structure called the **FP-Tree**.
2. Avoids repeated dataset scans by mining the FP-Tree recursively via **conditional pattern bases**.

### Algorithmic Steps
1. **Frequency Ordering**: Scan database once to find frequent 1-items. Sort items in descending order of frequency: $F$-list.
2. **FP-Tree Construction**:
   - Create root labeled `null`.
   - Scan database a second time. For each transaction, filter out infrequent items, sort remaining items by $F$-list, and insert them into the FP-tree.
   - Shared prefixes reuse existing nodes and increment their counts; distinct suffixes branch into new child nodes.
   - Maintain a **Header Table** with `node-links` chaining all nodes containing the same item.
3. **Recursive Mining**:
   - Traverse the Header Table bottom-up (least frequent item first).
   - Form prefix itemset $P \cup \{i\}$.
   - Follow `node-links` to traverse paths up to the root, forming the **Conditional Pattern Base**.
   - Construct a **Conditional FP-Tree** from the pattern base.
   - Recursively mine the conditional tree until empty.

### Complexity
- **Time**: Typically orders of magnitude faster than Apriori because it scans the dataset only twice and explores patterns depth-first.
- **Space**: Compact tree representation. Worst-case $\mathcal{O}(N \times M)$ if no paths are shared, but typically far smaller due to high prefix sharing.

---

## 3. ECLAT Algorithm (Zaki, 2000)

### Operating Principle
ECLAT (**Equivalence CLASS Transformation**) inverts the traditional horizontal database format (TID $\to$ Items) into a **vertical format** (Item $\to$ TID-set).

### Algorithmic Steps
1. **Vertical Inverted Index**: Build a mapping $\text{TID}(i) = \{t_1, t_2, \dots\}$ for each frequent item $i$.
2. **Support via Cardinality**: The exact support count of itemset $\{i\}$ is simply $|\text{TID}(i)|$.
3. **TID-Set Intersections**:
   - For items $i$ and $j$ sharing prefix $P$:
     $$\text{TID}(P \cup \{i, j\}) = \text{TID}(P \cup \{i\}) \cap \text{TID}(P \cup \{j\})$$
   - Support count is $|\text{TID}(P \cup \{i, j\})|$.
4. **Depth-First Search (DFS)**: Recurse on valid extensions whose intersection cardinality $\ge \text{min\_count}$.

### Complexity
- **Time**: Fast bitset / set intersections computed in memory without transaction rescans.
- **Space**: $\mathcal{O}(|I| \times N)$. TID-sets for very frequent items can consume considerable RAM when $N$ is very large (e.g. 500k+ records).

---

## 4. H-Mine Algorithm (Pei, Han, Lu, et al., ICDM 2001)

### Operating Principle
H-Mine (**Hyper-Structure Mining**) was designed to overcome the memory overhead of both FP-Tree recursion and vertical TID-set replication. It stores transactions in an array and threads **hyper-links** dynamically through transaction occurrences.

### Algorithmic Steps
1. **H-Struct Construction**:
   - Store filtered and sorted transactions as integer item vectors in main memory.
   - Create a Header Table for all frequent items.
   - Within each transaction, hyper-link nodes chain identical items across transactions.
2. **Projected Database via Hyper-Links**:
   - To project on item $i$, H-Mine traverses the linked list starting at $\text{Header}[i]$.
   - For each linked transaction, it examines the items occurring *after* item $i$.
   - It tallies local frequency counts of these subsequent items in the $i$-projected database.
3. **Dynamic Re-linking**:
   - Frequent extension items form projected header tables.
   - Transactions are dynamically re-linked into extension headers without copying transaction arrays.
4. **Recursive Mining**: Mines depth-first, recovering exact support counts without physical projection.

### Complexity
- **Time**: Scales well with database size; avoids tree building overhead and TID set allocation.
- **Space**: Minimal working memory beyond the single in-memory transaction array and hyper-link pointers.

---

## 5. Association Rule Metrics Formalism

Given a frequent itemset $I \subseteq \mathcal{I}$, an association rule is an implication of the form:
$$X \Rightarrow Y, \quad \text{where } X \subset I, \; Y = I \setminus X, \; X \cap Y = \emptyset, \; X \neq \emptyset, \; Y \neq \emptyset$$

### A. Support
Proportion of transactions containing both antecedent $X$ and consequent $Y$:
$$\text{Support}(X \Rightarrow Y) = \frac{\text{count}(X \cup Y)}{N}$$

### B. Confidence
Conditional probability of observing $Y$ given that $X$ is present:
$$\text{Confidence}(X \Rightarrow Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)} = \frac{\text{count}(X \cup Y)}{\text{count}(X)}$$

### C. Lift
Ratio of observed co-occurrence to the expected co-occurrence under statistical independence:
$$\text{Lift}(X \Rightarrow Y) = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)} = \frac{\text{Support}(X \cup Y)}{\text{Support}(X) \times \text{Support}(Y)}$$
- $\text{Lift} > 1$: Positive correlation (presence of $X$ increases likelihood of $Y$).
- $\text{Lift} = 1$: Independence.
- $\text{Lift} < 1$: Negative correlation / substitution effect.

### D. Leverage
Absolute difference between observed joint probability and independent product probability:
$$\text{Leverage}(X \Rightarrow Y) = \text{Support}(X \cup Y) - (\text{Support}(X) \times \text{Support}(Y))$$

### E. Conviction
Measures how strongly the rule would fail if $X$ and $Y$ were independent:
$$\text{Conviction}(X \Rightarrow Y) = \frac{1 - \text{Support}(Y)}{1 - \text{Confidence}(X \Rightarrow Y)}$$
When $\text{Confidence} = 1.0$, $\text{Conviction} = \infty$.
