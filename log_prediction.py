import warnings
import torch
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
import networkx as nx
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
from log_integrati import TrustSAGE

def nx_to_pyg(G):
    # --- Convert type string into numeric code ---
    NODE_TYPE_MAP = {
        'user': 0,
        'device': 1,
        'network': 2,
        'resource': 3
    }

    new_G = nx.DiGraph()
    new_G.add_nodes_from(G.nodes())
    new_G.add_edges_from(G.edges(data=True))

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "unknown")
        node_type_idx = NODE_TYPE_MAP.get(node_type, -1)

        trust = (
            attrs.get("trust") or
            attrs.get("trust_user") or
            attrs.get("trust_device") or
            attrs.get("trust_network") or
            attrs.get("impact_resource") or
            0.5
        )

        new_G.nodes[node]["node_type"] = node_type_idx
        new_G.nodes[node]["trust_value"] = float(trust)

    data = from_networkx(new_G)

    # ensure tensors are float / long
    data.node_type = data.node_type.long()
    data.trust_value = data.trust_value.float()
    data.weight = data.weight.float()

    num_node_types = 4  # user, device, network, resource
    node_type_one_hot = F.one_hot(data.node_type, num_classes=num_node_types).float()
    trust_col = data.trust_value.unsqueeze(1)  # shape [num_nodes,1]
    data.x = torch.cat([node_type_one_hot, trust_col], dim=1) 

    return data

def add_or_update_edge(G, n1, n2):
 
    if G.has_edge(n1, n2):
        # Incrementa il peso esistente
        G[n1][n2]["weight"] += 1
    else:
        # Crea nuovo arco con peso iniziale = 1
        G.add_edge(n1, n2, weight=1)

def add_request(G, user= None, device=None, network=None, resource=None, role=None, certificate=None):
    # nodi
    
    trust_device = 0.5
    trust_network = 0.5

    if role == "Administrator":
        trust_user = 0.8 
    else: 
        trust_user = 0.5  

    if user == None or user == "":
        user ="unknown_user"
        if role == "Administrator":
            trust_user -= 0.5
        else:
            trust_user -= 0.3
        trust_device -= 0.1
    
    if device == None or device == "":
        device ="unknown_device"
        trust_user -= 0.1
        trust_device -= 0.3
        trust_network -= 0.2
    
    if certificate is not None:
            trust_user += 0.2
            trust_device += 0.2
            trust_network += 0.2

    if user is not None and not G.has_node(user):
        G.add_node(user, type="user", trust = trust_user)

    if device is not None and not G.has_node(device):
        G.add_node(device, type="device", trust = trust_device)

    if network is not None and not G.has_node(network):
        G.add_node(network, type="network", trust = trust_network)
    
    if resource is not None and not G.has_node(resource):
        G.add_node(resource, type="resource", trust = 1)

    # archi
    if user and device:
        add_or_update_edge(G, user, device)

    if device and network:
        add_or_update_edge(G, device, network)

    if network and resource:
        add_or_update_edge(G, network, resource)

    return G

def predict_log(model, data, log, id_of_node, G, node_delta=0.2):
    
    
      
    u = id_of_node.get(log["user"])
    d = id_of_node.get(log["mac"])
    n = id_of_node.get(log["ip"])
    r = id_of_node.get(log["resource"])

    if None in (u, d, n, r):
        raise ValueError("Uno dei nodi del log non esiste nel grafo")

    triplet = torch.tensor([[u, d, n, r]], dtype=torch.long)

    
    model.eval()
    with torch.no_grad():
        logit = model(data, triplet)  # tensore
        prob = torch.sigmoid(logit).item()  # valore 0-1
        label = "DENY" if prob >= 0.5 else "ALLOW"

    print_trust_state(G, log["user"], log["mac"], log["ip"], log["resource"],title="PRIMA DELL'UPDATE")
    update_trust_graph(G, log["user"], log["mac"], log["ip"], log["resource"], label, node_delta=node_delta)
    save_trust_graph(G)
    G_after = load_trust_graph()
    print()
    print(
            f"[PREDICTION ST] "
            f"Decision: {label:<5} | "
            f"Suspicious prob: {prob:.4f} | "

        )
    
    print_trust_state(G_after, log["user"], log["mac"], log["ip"], log["resource"], title="DOPO L'UPDATE")
    return label, prob


def update_trust_graph(G, user, device, network, resource, label, node_delta=0.2):

    if label == "DENY":
        delta = -node_delta
    else:  # ALLOW
        delta = node_delta

    nodes = [user, device, network, resource]

    for node in nodes:
        if not G.has_node(node):
            continue

        old_trust = G.nodes[node].get("trust", 0.5)
        new_trust = old_trust + delta

        G.nodes[node]["trust"] = round(new_trust, 2)

def save_trust_graph(G, path="trust_graph.pkl"):
    with open(path, "wb") as f:
        pickle.dump(G, f)  

def load_trust_graph(path="trust_graph.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def print_trust_state(G, user, device, network, resource, title):
    print(f"\n--- {title} ---")
    for node in (user, device, network, resource):
        if G.has_node(node):
            t = G.nodes[node].get("trust")
            if isinstance(t, (int, float)):
                print(f"{node:25s} | trust = {t:.2f}")
            else:
                print(f"{node:25s} | trust = {t}")
        else:
            print(f"{node:25s} | NOT PRESENT")


if __name__ == "__main__":
    # Carica il modello
    checkpoint = torch.load("trustsage_best_model.pth", weights_only=False)
    # importa la definizione del modello

    model = TrustSAGE(
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["hidden_channels"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = checkpoint.get("threshold", 0.5)
    # Carica il grafo
    
    grp =  load_trust_graph()


    new_request = {
        "user": "vm-2008",
        "mac": "00:15:5D:2D:FC:FF",
        "ip": "192.168.3.99",
        "resource": "DataAuth",
        "role": "User",
        "certificate": "",
    }

    # new_request = {
    #     "user": "utente_nuovo",
    #     "mac": "dispositivo_nuovo",
    #     "ip": "192.168.3.99",
    #     "resource": "DataAuth",
    #     "role": "User",
    #     "certificate": "",
    # }
    
    # new_request = {
    #     "user": "vm-10",
    #     "mac": "00:15:5D:2D:FC:25",
    #     "ip": "192.168.3.120",
    #     "resource": "Data",
    #     "role": "Administrator",
    #     "certificate": "A8F8213186"
    # } 
    
    # new_request = {
    #     "user": "nuovo_utente",
    #     "mac": "nuovo_dispositivo",
    #     "ip": "192.168.3.120",
    #     "resource": "Data",
    #     "role": "Administrator",
    #     "certificate": "A8F821318660A91949753F4A36EF6542095B96B5"
    # }   

    print ("===============")
    print ("Updating graph....")
    updated_graph = add_request(
            grp, user=new_request["user"], device=new_request["mac"],
            network=new_request["ip"], resource=new_request["resource"],
            role=new_request["role"], certificate=new_request["certificate"]
        )
        
    u_id_of_node = {node: i for i, node in enumerate(updated_graph.nodes())}
       
    data = nx_to_pyg (grp)
        
    print ("===============")
    print ("Testing new log incoming....")
   
    label, prob = predict_log(model, data, new_request, u_id_of_node, updated_graph)

    


