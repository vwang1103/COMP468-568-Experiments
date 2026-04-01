import random

def generate_graph_edges_txt(filename, num_nodes, avg_out_degree, seed=1):
    random.seed(seed)

    with open(filename, "w") as f:
        for u in range(num_nodes):
            neighbors = set()
            while len(neighbors) < avg_out_degree:
                v = random.randrange(num_nodes)
                if v != u:
                    neighbors.add(v)

            for v in neighbors:
                f.write(f"{u} {v}\n")

    print(f"Saved {filename}")
    print(f"num_nodes={num_nodes}, avg_out_degree={avg_out_degree}, approx_nnz={num_nodes * avg_out_degree}")

if __name__ == "__main__":
    generate_graph_edges_txt("graph_edges.txt", num_nodes=1024, avg_out_degree=32)