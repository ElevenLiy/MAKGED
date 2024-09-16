We use two real-world knowledge graph datasets: FB15K and WN18RR. In each dataset, we simulate realistic errors by replacing entities and relations with the highest semantic similarity, with approximately 30\% of the data being erroneous. FB15K is derived from Freebase and contains a rich set of entities and relations, while WN18RR is a subset of WordNet with corrected inverse relations, increasing the dataset's complexity. 


The TXT format file contains triples where we used similarity matching to find the most similar entities and relations. A 0 at the end of a triple indicates no error was inserted. A 1 indicates that the head entity was replaced, a 2 indicates that the relation was replaced, and a 3 indicates that the tail entity was replaced.
