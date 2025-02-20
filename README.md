

# MAKGED: Multi-Agent Framework for Knowledge Graph Error Detection
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2501.15791)
[![GitHub Stars](https://img.shields.io/github/stars/kse-ElEvEn/MAKGED?style=social)](https://github.com/kse-ElEvEn/MAKGED)

<p align="center">
  <img width="972" alt="image" src="https://github.com/user-attachments/assets/ba8afcb0-a0f0-4622-8478-c0bd80884dc5">
</p>

## 📌 Overview

**MAKGED** is the first multi-agent framework for collaborative error detection in knowledge graphs, addressing two critical challenges in KG quality management:
1. **Multi-Perspective Analysis**: Overcomes single-view limitations through bidirectional subgraph analysis
2. **Transparent Decision-Making**: Implements explainable error detection via structured agent discussions

Key innovations:
- 🎯 **4 Specialized Agents**: Head/Tail × Forward/Backward agent architecture
- 🔍 **Hybrid Embeddings**: Combines GCN-based structural features with LLM semantic features
- 🤖 **LLM-Powered Collaboration**: Implements 3-round discussion protocol with tiebreaker mechanism
- 🏭 **Industrial Proven**: Validated with China Mobile's KGs


## 🧠 Framework Architecture

### Core Components
1. **Bidirectional Subgraph Construction**
   - Head_Forward/Backward Subgraphs
   - Tail_Forward/Backward Subgraphs
   
2. **Hybrid Embedding Generator**
   ```mermaid
   graph TD
     A[Subgraph Structure] --> B[3-Layer GCN]
     C[Triple Text] --> D[Llama-2 Embedding]
     B --> E[Concatenation Layer]
     D --> E
     E --> F[Unified Representation]
   ```

3. **Multi-Agent Discussion Protocol**
   - Phase 1: Independent Analysis
   - Phase 2: 3-Round Discussion
   - Phase 3: Summarizer Voting (for ties)

## 📊 Datasets

| Dataset    | Triples  | Entities | Relations | Error Rate |
|------------|----------|----------|-----------|------------|
| FB15K      | 44,000  | 14,541   | 237     | 30.2%      |
| WN18RR     | 33,134   | 40,943   | 11        | 30.7%      |


## 📈 Benchmark Results

### Performance Comparison (FB15K)

| Models                             | **FB15K** Accuracy | **FB15K** F1-Score | **FB15K** Precision | **FB15K** Recall | **WN18RR** Accuracy | **WN18RR** F1-Score | **WN18RR** Precision | **WN18RR** Recall |
|------------------------------------|--------------------|--------------------|---------------------|------------------|---------------------|---------------------|----------------------|-------------------|
| **Embedding-Based Methods**        |                    |                    |                     |                  |                     |                     |                      |                   |
| TransE                             | 0.6373             | 0.6312             | 0.6410              | 0.6531           | 0.3813              | 0.2927              | 0.6255               | 0.5083            |
| DistMult                           | 0.5938             | 0.5132             | 0.5261              | 0.5204           | 0.6401              | 0.5157              | 0.5965               | 0.5449            |
| ComplEx                            | 0.6268             | 0.4781             | 0.5413              | 0.5172           | 0.6414              | 0.4450              | 0.6464               | 0.5217            |
| CAGED                              | 0.6091             | 0.4574             | 0.5028              | 0.4552           | 0.6544              | 0.5064              | 0.5532               | 0.5013            |
| KGTtm                              | 0.6828             | 0.4078             | 0.6172              | 0.3045           | 0.6911              | 0.4487              | 0.6589               | 0.3402            |
| **PLM-based Methods**              |                    |                    |                     |                  |                     |                     |                      |                   |
| KG-BERT                            | 0.7675             | 0.6280             | 0.7371              | 0.5470           | 0.8162              | 0.7222              | 0.8177               | 0.6468            |
| StAR                               | 0.7350             | 0.6017             | 0.6900              | 0.5420           | 0.7012              | 0.6100              | 0.6572               | 0.5645            |
| CSProm-KG                          | 0.7078             | 0.5509             | 0.6139              | 0.4997           | 0.7116              | 0.6025              | 0.6138               | 0.4997            |
| **Contrastive Learning-based Methods** |                |                    |                     |                  |                     |                     |                      |                   |
| SeSICL                             | 0.5950             | 0.4600             | 0.5513              | 0.5172           | 0.5050              | 0.4073              | 0.4421               | 0.5711            |
| CCA                                | 0.7456             | 0.6810             | 0.7123              | 0.6537           | 0.7621              | 0.7134              | 0.7568               | 0.6912            |
| **LLM-based Methods**              |                    |                    |                     |                  |                     |                     |                      |                   |
| Llama2                             | 0.7420             | 0.6010             | 0.7250              | 0.6851           | 0.7100              | 0.6271              | 0.7021               | 0.6344            |
| GPT-3.5                            | 0.7445             | 0.6117             | 0.7185              | 0.6555           | 0.7603              | 0.7496              | 0.7120               | 0.6260            |
| Llama3                             | 0.7558             | 0.6264             | 0.7357              | 0.7148           | 0.7654              | 0.7522              | 0.7185               | 0.6327            |
| **Our Methods**                    |                    |                    |                     |                  |                     |                     |                      |                   |
| MAKGED                             | **0.7748**         | **0.7367**         | **0.7686**           | **0.7252**        | **0.8283**          | **0.7909**          | **0.8832**            | **0.7704**         |

### Industrial Case Study
<p align="center">
  <img width="1087" alt="image" src="https://github.com/user-attachments/assets/2c290faa-aba0-48df-87e8-0b7984f6aa2b">
</p>


## 📚 Citation
```bibtex
@article{li2025harnessing,
  title={Harnessing Diverse Perspectives: A Multi-Agent Framework for Enhanced Error Detection in Knowledge Graphs},
  author={Li, Yu and Huang, Yi and Qi, Guilin and Feng, Junlan and Hu, Nan and Zhai, Songlin and Xue, Haohan and Chen, Yongrui and Shen, Ruoyan and Wu, Tongtong},
  journal={arXiv preprint arXiv:2501.15791},
  year={2025}
}
```

## 📧 Contact
For technical inquiries:  
[Yu Li](mailto:yuli_11@seu.edu.cn) - Southeast University  

---

**[⬆ Back to Top](#makged-multi-agent-framework-for-knowledge-graph-error-detection)**
