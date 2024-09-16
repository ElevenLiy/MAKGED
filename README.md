# MAKGED

Knowledge graphs (KGs) are widely used in natural language processing tasks, making error detection crucial to maintain quality and performance. Current methods, including rule-based and embedding-based approaches, struggle to generalize across different KGs or fully utilize subgraph information, leading to unsatisfactory false detection results.
In this paper, we propose MAKGED, a novel framework for KG error detection that utilizes multiple large language models (LLMs) in a collaborative setting. Subgraph embeddings are generated using a graph convolutional network (GCN) and concatenated with LLM-generated query embeddings to ensure effective comparison of semantic and structural information. Four agents trained on different subgraph strategies analyze the triples and collaborate through multi-round discussions, improving both detection accuracy and transparency.
Extensive experiments on FB15K and WN18RR demonstrate that MAKGED outperforms state-of-the-art methods, improving the accuracy and robustness of KG evaluation. 

<img width="972" alt="image" src="https://github.com/user-attachments/assets/ba8afcb0-a0f0-4622-8478-c0bd80884dc5">


<img width="1087" alt="image" src="https://github.com/user-attachments/assets/2c290faa-aba0-48df-87e8-0b7984f6aa2b">
