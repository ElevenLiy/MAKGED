import json
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch_scatter import scatter_mean
import logging
import os

def setup_logging():
    logger = logging.getLogger('gcn_training')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler('fb15k_training1.log')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger

logger = setup_logging()

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# 将三元组字符串转换为图结构
def parse_triplets(triplets, max_nodes=51):
    entities = {}
    edge_index = []
    for triplet in triplets:
        head, relation, tail = triplet.split(', ')
        if head not in entities:
            entities[head] = len(entities)
        if tail not in entities:
            entities[tail] = len(entities)
        edge_index.append([entities[head], entities[tail]])

    num_nodes = len(entities)
    if num_nodes < max_nodes:

        x = torch.eye(max_nodes)
        for i in range(num_nodes, max_nodes):
            x[i] = torch.zeros(max_nodes)
    else:
        x = torch.eye(num_nodes)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0),
                                                                                                            dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def prepare_data(json_path, max_nodes=51):
    json_data = load_data(json_path)
    dataset = []
    for item in json_data:
        label = torch.tensor([item['label']], dtype=torch.long)
        for key in ['Head entity as head', 'Head entity as tail', 'Tail entity as head', 'Tail entity as tail']:
            triplets = item[key]
            if triplets:
                graph = parse_triplets(triplets, max_nodes)
                graph.y = label
                dataset.append(graph)
            else:
                # 处理空子图的情况
                graph = Data(x=torch.zeros(max_nodes, max_nodes), edge_index=torch.empty((2, 0), dtype=torch.long), y=label)
                dataset.append(graph)
    return dataset


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = scatter_mean(x, batch, dim=0)
        embedding = x
        logits = self.classifier(embedding)
        return embedding, logits


def train_model(train_data, dev_data, model, optimizer, scheduler, logger):
    loader = DataLoader(train_data, batch_size=64, shuffle=True)
    best_val_loss = float('inf')
    for epoch in range(100):
        model.train()
        total_train_loss = 0
        for data in tqdm(loader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()
            _, logits = model(data)
            loss = F.cross_entropy(logits, data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(loader)
        logger.info(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}')
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}')

        avg_val_loss = evaluate_model(dev_data, model, logger)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),
                       'fb15k_best_model1.pth')
            print(f'New best model saved with Validation Loss: {best_val_loss:.4f}')
        else:
            print(f'No improvement in Validation Loss: {avg_val_loss:.4f} vs Best: {best_val_loss:.4f}')


def evaluate_model(dev_data, model, logger):
    loader = DataLoader(dev_data, batch_size=64, shuffle=False)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            _, logits = model(data)
            loss = F.cross_entropy(logits, data.y.view(-1))
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == data.y.view(-1)).sum().item()
            total += data.y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    logger.info(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss


def main():
    train_data = prepare_data('fb15k_train.json')
    dev_data = prepare_data('fb15k_dev.json')
    model = GCN(input_dim=train_data[0].num_features, hidden_dim=128, embedding_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    logger = setup_logging()
    train_model(train_data, dev_data, model, optimizer, scheduler, logger)


if __name__ == '__main__':
    main()
