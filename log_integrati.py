import pickle
import warnings
import torch
import random
import numpy as np

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(42)

warnings.filterwarnings("ignore", category=UserWarning)

from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import json
import math
import copy 
import pandas as pd
from datetime import datetime, timedelta
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GraphNorm, SAGEConv, GATConv
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, recall_score, precision_score

ROLE_WEIGHTS = {
    'Administrator': 0.8,
    'User': 0.5,
}


def parse_time(ts):
    from datetime import datetime
    try:
        # con microsecondi
        return datetime.strptime(ts, "%d-%b-%Y %H:%M:%S.%f")
    except:
        try:
            # senza microsecondi
            return datetime.strptime(ts, "%d-%b-%Y %H:%M:%S")
        except:
            # fallback ISO
            return datetime.fromisoformat(ts)

def within_window(t1, t2, minutes=2):
    """Controlla se due timestamp sono entro +/- 2 minuti."""
    return abs(t1 - t2) <= timedelta(minutes=minutes)

def correlate_logs(events):
    """
    events = lista di dict, come quelli che hai fornito.
    Ritorna lista di eventi aggregati.
    """

    # Separa p0f, applicativi, captive portal
    p0f_by_ip = defaultdict(list)
    app_events = []
    captive_events = []

    for e in events:
        src = e.get("source")
        msg = e.get("message", {})
        ts = parse_time(e["time"])

        if src == "p0f":
            for ip, details in msg.items():
                details_dict = details if isinstance(details, dict) else {"info": details}
                p0f_by_ip[ip].append({
                    "time": ts,
                    **details_dict
                })  

        elif src in ("Blazor:2.0.0.2:AuthAudit", "Blazor:2.0.0.2:ApplicationAudit"):
            app_events.append({
                "time": ts,
                "event": msg,
                "source": src
            })

            

        elif src == "captiveportal":
            captive_events.append({
                "time": ts,
                "event": msg,
                "source": src
            })


    # Raggruppo eventi applicativi per utente
    user_buckets = defaultdict(list)
    captive_buckets = defaultdict(list)
    for e in app_events:
        user = e["event"].get("User", "UNKNOWN")
        ip = e["event"].get("IP")
        if ip:
            user_buckets[user].append(e)

    # Aggiungo utenti dal captive portal
    for e in captive_events:
        ip = e["event"].get("ip")
        if ip:
            captive_buckets[ip].append(e)
    

    output = []

 
    for user, uevents in user_buckets.items():

        # Raggruppa per IP dello stesso utente
        ips = defaultdict(list)
        for e in uevents:
            ip = e["event"].get("IP")
            if ip:
                ips[ip].append(e)

        for ip, events_same_ip in ips.items():

            # Trova finestre temporali min/max degli eventi app
            sorted_events = sorted(events_same_ip, key=lambda x: x["time"])
            t_start = sorted_events[0]["time"]
            t_end = sorted_events[-1]["time"]

            # Cerca nei p0f match entro +/- 2 minuti
            correlated_p0f = []
            for p in p0f_by_ip.get(ip, []):
                if within_window(p["time"], t_start) or within_window(p["time"], t_end):
                    correlated_p0f.append(p)
            
            # Preparo output consolidato
            out = {
                "User": user,
                "IP": ip,
                "TimeWindow": {
                    "start": t_start.isoformat(),
                    "end": t_end.isoformat()
                },
                "Events": [
                    {
                        "source": e["source"],
                        "time": e["time"].isoformat(),
                        "event": e["event"]
                    }
                    for e in sorted_events
                ],
                "p0f": correlated_p0f
            }

            if ip in captive_buckets:
                 out["captive"] = [
            {
                "source": e["source"],
                "time": e["time"].isoformat(),
                "event": e["event"]
            }
            for e in captive_buckets[ip]
        ]

            output.append(out)

    return output

def create_logs(grouped_logs):
    
    ip_mac_map = {}
    mac_hostname_map = {}

    for entry in grouped_logs:
        ip = entry.get("IP")
        for p in entry.get("p0f", []):
            mac = p.get("mac")
            hostname = p.get("hostname")
            if ip and mac:
                ip_mac_map[ip] = mac
            if mac and hostname and hostname != "unknown":
                mac_hostname_map[mac] = hostname
    

    result_logs = []
    for entry in grouped_logs:
        user = entry["User"]
        ip = entry["IP"]
        events = entry["Events"]
        p0f = entry["p0f"]
        


        for event in events:
            event_time = datetime.strptime(event["time"], "%Y-%m-%dT%H:%M:%S") if isinstance(event["time"], str) else event["time"]
            event_source = event["source"]
            event_type = event["event"]["EventType"]
            suspicious = event["event"]["Suspicious"]
            # resource = event["event"]["ResourceContext"]["database"]
            certificate = event["event"]["ClientCertificate"]
            user_context = event.get("event", {}).get("UserContext")

            resource_context = event["event"].get("ResourceContext")

            if isinstance(resource_context, dict) and "database" in resource_context:
                resource = resource_context["database"]
            else:
                resource = "N/A"
                


            if user_context and user_context.get("roles"):
                role = user_context["roles"][-1]  # ultimo valore della lista
            else:
                role = "User"

            status_code = event["event"].get("HTTPStatusCode", None)
            
            if status_code is None and not suspicious:
                status_code = 200
            elif status_code is None and suspicious:
                status_code = 401 
            else:
                pass
 

            # Trova l'entry p0f con il time precedente all'evento
            closest_p0f_record = None
            closest_time_diff = None

            for p0f_entry in p0f:
                p0f_time = p0f_entry["time"]
                # Se il time è una stringa, convertila in datetime
                if isinstance(p0f_time, str):
                    p0f_time = datetime.strptime(p0f_time, "%Y-%m-%d %H:%M:%S")
                
                # Controlla se p0f_time è prima dell'evento
                if p0f_time < event_time:
                    # Calcola la differenza di tempo tra l'evento e il p0f
                    time_diff = event_time - p0f_time
                    
                    # Se è il primo p0f o il più vicino, aggiorna
                    if closest_time_diff is None or time_diff < closest_time_diff:
                        closest_time_diff = time_diff
                        closest_p0f_record = p0f_entry
            
            if closest_p0f_record:
                mac = closest_p0f_record["mac"]
                hostname = closest_p0f_record["hostname"]
               
            else:
                mac = None
                hostname = None

            if mac is None:
                captive_events = entry.get("captive")
                closest_captive_record = None
                closest_cap_diff = None
                if captive_events:
                    for cap in captive_events:
                        cap_time = cap["time"]
                        if isinstance(cap_time, str):
                            cap_time = datetime.strptime(cap_time, "%Y-%m-%dT%H:%M:%S")

                        if cap_time < event_time:
                            time_diff = event_time - cap_time

                            if closest_cap_diff is None or time_diff < closest_cap_diff:
                                closest_cap_diff = time_diff
                                closest_captive_record = cap

                    if closest_captive_record:
                        mac = (closest_captive_record["event"].get("mac")).upper()
                
            if ip and mac is None:
                mac = ip_mac_map.get(ip)

            if mac and hostname in (None, "unknown"):
                hostname = mac_hostname_map.get(mac)   

            if mac and mac.startswith("INCOMPLETE_"):
                mac = None    

            # Crea un log con le informazioni richieste
            log = {
                "user": user,
                "ip": ip,
                "role": role,
                "mac": mac,
                "hostname": hostname,
                "certificate": certificate,
                "source": event_source,
                "event_time": event_time,
                "event_type": event_type,
                "resource": resource,
                "suspicious": suspicious,
                "status_code": status_code
            }
            
            if resource and resource != "N/A":
                result_logs.append(log)

    return result_logs




def build_trust_graph(correlated_logs):
    G = nx.DiGraph()

    user_device = defaultdict(int)
    device_network = defaultdict(int)
    network_resource = defaultdict(int)

    user_scores = {}
    device_weights = {}
    network_weights = {}


    for log in correlated_logs:
        user = log.get('user') or 'unknown_user'
        device = log.get('mac') or 'unknown_device'
        network = log.get('ip') 
        resource = log.get('resource') 
        certificate = log.get('certificate')
        decision = log.get('suspicious')
        role = log.get('role')
        timestamp = log.get('event_time')

        # increment edge counters
        user_device[(user, device)] += 1
        device_network[(device, network)] += 1
        network_resource[(network, resource)] += 1
        
        delta_suspicious = -0.7
        delta_clean = +0.2

        raw_ts = timestamp

        if isinstance(raw_ts, datetime):
            event_time = raw_ts
        elif isinstance(raw_ts, str):
            try:
                event_time = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
            except:
                event_time = datetime.fromisoformat(raw_ts.replace(" ", "T"))
        elif isinstance(raw_ts, (int, float)):
            event_time = datetime.utcfromtimestamp(raw_ts)
        else:
            print("Timestamp non valido:", raw_ts, type(raw_ts))
           
                
        now = datetime.utcnow()
        age_seconds = (now - event_time).total_seconds()

        lambda_ = 0.000001 
        time_weight = math.exp(-lambda_ * age_seconds)

        start_trust = ROLE_WEIGHTS.get(role)
        base_weight = 0.5
        base_weightn = 0.5

        if user == "unknown_user":
            start_trust -= 0.6
            base_weight -= 0.3

        if device == "unknown_device":
            start_trust -= 0.3
            base_weight -= 0.4
            base_weightn -= 0.2
        
        if certificate is not None:
            start_trust += 0.3
            base_weight += 0.3
            base_weightn += 0.3

        if user not in user_scores and decision:
            user_scores[user] = start_trust + (delta_suspicious * time_weight)
        elif user not in user_scores and not decision:
            user_scores[user] = start_trust + (delta_clean * time_weight)
        elif decision:
            user_scores[user] += delta_suspicious * time_weight * start_trust
        else:
            user_scores[user] += delta_clean * time_weight * start_trust

        if device not in device_weights and decision:
            device_weights[device] = base_weight + (delta_suspicious * time_weight)
        elif device not in device_weights and not decision:
            device_weights[device] = base_weight + (delta_clean * time_weight)
        elif decision:
            device_weights[device] += delta_suspicious * time_weight
        else:
            device_weights[device] += delta_clean * time_weight
        
        if network not in network_weights and decision:
            network_weights[network] = base_weightn + (delta_suspicious * time_weight)
        elif network not in network_weights and not decision:
            network_weights[network] = base_weightn + (delta_clean * time_weight)
        elif decision:
            network_weights[network] += delta_suspicious * time_weight
        else:
            network_weights[network] += delta_clean * time_weight
        


    # add user nodes
    for user, scores in user_scores.items():
        G.add_node(user, type='user', trust = round(scores, 2))

    # add device nodes
    for device, weight in device_weights.items():
        G.add_node(device, type='device', trust = round(weight, 2))

    # add network nodes
    for network, weight in network_weights.items():
        G.add_node(network, type='network', trust = round(weight, 2))

    # add resource nodes
    for _, resource in network_resource.keys():
        G.add_node(resource, type='resource', trust=1.0)

    # add edges with weights
    for (u, d), freq in user_device.items():
        G.add_edge(u, d, weight=freq)

    for (d, n), freq in device_network.items():
        G.add_edge(d, n, weight=freq)

    for (n, r), freq in network_resource.items():
        G.add_edge(n, r, weight=freq)

    return G

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

def build_training_samples(logs, id_of_node):
    X_pairs = []
    y = []

    for log in logs:
        user = log.get("user") or "unknown_user" 
        device = log.get("mac") or "unknown_device"
        network = log.get("ip")
        resource = log.get("resource")
        decision = log.get("suspicious")
        # ricaviamo gli ID numerici dei nodi
        u = id_of_node[user]
        d = id_of_node[device]
        n = id_of_node[network]
        r = id_of_node[resource]

        # tripletta di riferimento
        X_pairs.append([u, d, n, r])

        y.append(1 if decision else 0)

    return torch.tensor(X_pairs, dtype=torch.long), torch.tensor(y, dtype=torch.float)


# def add_or_update_edge(G, n1, n2):
 
#     if G.has_edge(n1, n2):
#         # Incrementa il peso esistente
#         G[n1][n2]["weight"] += 1
#     else:
#         # Crea nuovo arco con peso iniziale = 1
#         G.add_edge(n1, n2, weight=1)

# def add_request(G, user= None, device=None, network=None, resource=None, role=None, certificate=None):
#     # nodi
    
#     trust_device = 0.5
#     trust_network = 0.5

#     if role == "Administrator":
#         trust_user = 0.8 
#     else: 
#         trust_user = 0.5  

#     if user == None or user == "":
#         user ="unknown_user"
#         if role == "Administrator":
#             trust_user -= 0.5
#         else:
#             trust_user -= 0.3
#         trust_device -= 0.1
    
#     if device == None or device == "":
#         device ="unknown_device"
#         trust_user -= 0.1
#         trust_device -= 0.3
#         trust_network -= 0.2
    
#     if certificate is not None:
#             trust_user += 0.2
#             trust_device += 0.2
#             trust_network += 0.2

#     if user is not None and not G.has_node(user):
#         G.add_node(user, type="user", trust = trust_user)

#     if device is not None and not G.has_node(device):
#         G.add_node(device, type="device", trust = trust_device)

#     if network is not None and not G.has_node(network):
#         G.add_node(network, type="network", trust = trust_network)
    
#     if resource is not None and not G.has_node(resource):
#         G.add_node(resource, type="resource", trust = 1)

#     # archi
#     if user and device:
#         add_or_update_edge(G, user, device)

#     if device and network:
#         add_or_update_edge(G, device, network)

#     if network and resource:
#         add_or_update_edge(G, network, resource)

#     return G


def get_edge_weights_batch(edge_index, edge_weight, sources, targets):
    
    edge_weight = edge_weight / (edge_weight.max() + 1e-8)
    src = edge_index[0]
    dst = edge_index[1]
    mask = (sources.unsqueeze(1) == src.unsqueeze(0)) & \
           (targets.unsqueeze(1) == dst.unsqueeze(0))
    weights = mask.float() @ edge_weight
    return weights.unsqueeze(1)


class TrustSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = GraphNorm(hidden_channels) # stabilizing training
        self.bn2 = GraphNorm(hidden_channels)
        self.edge_embed = nn.Linear(1, hidden_channels)
        self.concat_norm = nn.LayerNorm(hidden_channels*7)
        self.fc = nn.Linear(hidden_channels*7, 1)  # concat user-device-network-resource
        self.dropout = nn.Dropout(0.3) 
        self.dropout_combined = nn.Dropout(0.3)

    def forward(self, data, triplets):
        x, edge_index, edge_weight = data.x, data.edge_index, data.weight
        # GNN propagation
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x) 
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x) 
        x = F.relu(x)
        x = self.dropout(x)

        # Estrai embedding dei nodi coinvolti nelle richieste
        u = triplets[:, 0]
        d = triplets[:, 1]
        n = triplets[:, 2]
        r = triplets[:, 3]

        u_embed = x[u]
        d_embed = x[d]
        n_embed = x[n]
        r_embed = x[r]

        w_ud = get_edge_weights_batch(edge_index, edge_weight, u, d)
        w_dn = get_edge_weights_batch(edge_index, edge_weight, d, n)
        w_nr = get_edge_weights_batch(edge_index, edge_weight, n, r)

        w_ud_embed = self.edge_embed(w_ud)
        w_dn_embed = self.edge_embed(w_dn)
        w_nr_embed = self.edge_embed(w_nr)

        # Concatena embedding
        combined = torch.cat([u_embed, d_embed, n_embed, r_embed, w_ud_embed, w_dn_embed,w_nr_embed], dim=1)
        combined = self.concat_norm(combined)
        combined = self.dropout_combined(combined)
        out = self.fc(combined)
        return out.squeeze()



def train(model, data, X_train, y_train, X_test=None, y_test=None, epochs=50, lr=0.0003):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # num_pos = y_train.sum()            
    # num_neg = len(y_train) - num_pos   
    # pos_weight = torch.clamp(torch.tensor(num_neg / num_pos), max=10.0)

    criterion = nn.BCEWithLogitsLoss() #binary output
    
    #early model selection
    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data,  X_train)
        loss = criterion(out, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step() #updated weights

        if epoch % 10 == 0 and X_test is not None:
            model.eval()
            with torch.no_grad():
                logits = model(data, X_test)
                probs = torch.sigmoid(logits)
                pred_labels = (probs >= 0.5).float()
                f1 = f1_score(y_test, pred_labels)
                acc = (pred_labels == y_test).sum() / y_test.size(0)
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}, F1: {f1:.4f}")
                if f1 > best_f1: 
                    best_f1 = f1
                    best_state = model.state_dict()
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state) 


def kfold_train(model_class, data, X, y, k=5, epochs=50,lr=0.0003):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    f1_scores = []
    acc_scores = []
    precision_scores = []
    recall_scores = []
    
    best_f1 = -1
    best_model = None
    best_threshold = 0.5

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== Fold {fold+1}/{k} =====")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # nuovo modello per ogni fold
        model = model_class()

        train(
            model=model,
            data=data,
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
            epochs=epochs,
            lr=lr
        )

        # valutazione finale sul fold
        model.eval()
        with torch.no_grad():
            logits = model(data, X_val)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_val_cpu = y_val.cpu().numpy()

            precision_arr, recall_arr, thresholds = precision_recall_curve(y_val_cpu, probs)
            f1_per_threshold = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
            best_idx = f1_per_threshold.argmax()
            fold_best_threshold = thresholds[best_idx]
            preds = (probs >= fold_best_threshold).astype(float)

            acc = (preds == y_val).float().mean()
            f1 = f1_score(y_val_cpu, preds, pos_label=1)
            recall = recall_score(y_val_cpu, preds, pos_label=1)
            precision = precision_score(y_val_cpu, preds, pos_label=1)

            # Salvataggio dei risultati per ogni fold
            recall_scores.append(recall)
            precision_scores.append(precision)
            f1_scores.append(f1)
            acc_scores.append(acc)

            print(f"Fold {fold+1} → Acc: {acc:.4f}, F1 (1): {f1:.4f}, Precision (1): {precision:.4f}, Recall (1): {recall:.4f}")

            cm = confusion_matrix(y_val.cpu().numpy(), preds, labels=[0, 1])

            cm_df = pd.DataFrame(
                cm,
                index=["True 0", "True 1"],
                columns=["Pred 0", "Pred 1"]
            )

            print("Matrice di Confusione:")
            print(cm_df)
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model)
                best_threshold = fold_best_threshold  # cloniamo il modello
        

    return best_model, best_threshold
    




if __name__ == "__main__":
    # Carica log da file:
    with open("logs.json") as f:
        events = json.load(f)

    grouped_logs = correlate_logs(events)
    result = create_logs (grouped_logs)

    # with open("final_output_correlated.json", "w", encoding="utf-8") as f:
    #    json.dump(result, f, indent=4, default=str, ensure_ascii=False)

    grp = build_trust_graph(result)
 
    id_of_node = {node: i for i, node in enumerate(grp.nodes())}


    
    X_pairs, y = build_training_samples(result, id_of_node)
    
    data = nx_to_pyg (grp)
    
    
    model, threshold = kfold_train(model_class= lambda: TrustSAGE(in_channels=data.x.shape[1],hidden_channels=64), data=data, X=X_pairs, y=y, k=5, epochs=100, lr=0.0003)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": data.x.shape[1],
        "hidden_channels": 64,
        "threshold": threshold
    }, "trustsage_best_model.pth")

    with open("trust_graph.pkl", "wb") as f:
        pickle.dump(grp, f)

    # print ("-----------------------------------------------------------")
    # print ("========= FINAL TRAINING =========")
    # X_train, X_test, y_train, y_test = train_test_split(X_pairs, y, test_size=0.2, random_state=42, stratify=y)
    # model = TrustSAGE(in_channels=data.x.shape[1], hidden_channels=64)
    # train(model, data,  X_train, y_train, X_test, y_test, epochs= 100, lr=0.0005)
   
   
   
    # new_request = {
    # "user": "mario",
    # "mac": "00:15:5D:2D:FC:25",
    # "ip": "192.168.3.120",
    # "resource": "Data",
    # "role": "Administrator",
    # "certificate": "A8F821318660A91949753F4A36EF6542095B96B5"
    # }

    # print ("===============")
    # print ("Updating graph....")
    # updated_graph = add_request(
    #     grp, user=new_request["user"], device=new_request["mac"],
    #     network=new_request["ip"], resource=new_request["resource"],
    #     role=new_request["role"], certificate=new_request["certificate"]
    # )
    
    # u_id_of_node = {node: i for i, node in enumerate(updated_graph.nodes())}
    
    
    # updated_data = nx_to_pyg (updated_graph)
    
    # print ("===============")
    # print ("Testing new log incoming....")
    # label, confidence = predict_log(model, updated_data, new_request, u_id_of_node)
    # print(f"Decisione: {label}, Probabilità: {confidence:.4f}")






    # print("\n=== GRAFO AGGIORNATO ===")

    # print("=== NODI ===")
    # for node, attrs in grp.nodes(data=True):
    #     print(f"{node}: {attrs}")

    # print("\n=== ARCHI ===")
    # for u, v, attrs in grp.edges(data=True):
    #     print(f"{u} -> {v}: {attrs}")

    # pos = nx.kamada_kawai_layout(grp, scale=15)
    # # Colori dei nodi basati sul tipo
    # node_colors = []
    # for n, attr in grp.nodes(data=True):
    #     if attr['type'] == 'user':
    #         node_colors.append('skyblue')
    #     elif attr['type'] == 'device':
    #         node_colors.append('orange')
    #     elif attr['type'] == 'network':
    #         node_colors.append('green')
    #     elif attr['type'] == 'resource':
    #         node_colors.append('red')

    # # Dimensione dei nodi proporzionale al trust (con scala)
    # node_sizes = [max(30, abs(attr['trust']) / 10) for n, attr in grp.nodes(data=True)]

    # # Disegniamo il grafo
    # plt.figure(figsize=(24, 20))
    # nx.draw_networkx_nodes(grp, pos, node_color=node_colors, node_size=node_sizes)
    # nx.draw_networkx_edges(grp, pos, edge_color='gray', width=1)  # spessore uniforme
    # nx.draw_networkx_labels(grp, pos, font_size=8)

    # # Aggiungi etichette dei pesi sugli archi
    # edge_labels = {(u, v): d['weight'] for u, v, d in grp.edges(data=True)}
    # nx.draw_networkx_edge_labels(grp, pos, edge_labels=edge_labels, font_size=6)

    # plt.title("Grafo con nodi colorati per tipo, dimensione trust e peso archi etichettato")
    # plt.axis('off')
    # plt.show()