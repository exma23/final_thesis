#include <vector>
#include <map>
#include <iostream>
#include <tuple>
#include <unordered_set>

using std::get;
using edge_tuple = std::tuple<std::tuple<int, int>, std::tuple<int, int>>;
using Edge = std::tuple<int, int>;

int & get_head(Edge &edge) {return get<0> (edge);}
int & get_tail(Edge &edge) {return get<1> (edge);}

class Fnode{
    public:
    int index;
    bool leaf;
    std::vector<Fnode*> links;

    Fnode() {
        links = std::vector<Fnode*> (3, nullptr); 
    }

    Fnode(int idx, bool l){
        index = idx;
        leaf = l;
        links = std::vector<Fnode*> (3, nullptr); 
    }

    void add_link(Fnode & node){
        int i = 0;
        bool assigned = false;
        while(not assigned){
            if(links[i] == nullptr) {links[i] = &node; assigned = true;}
            else i++;

        }
    } 

};


class Ftree{
    public:
    int n_taxa;
    int m;
    double * d;
    double tree_length;
    bool nni_repetition;

    std::vector<Edge> edges;
    std::map<Edge, int> edge_idxs;

    std::vector<Edge> subtrees;
    std::vector<double> subtrees_n_nodes;
    std::vector<double> subtrees_max_length;
    std::vector<std::vector<double>> subtree_distance;
    std::vector<double> subtrees_total_length;

    std::map<Edge, double> branch_length;
    std::vector<std::vector<short>> T;
    std::map<int, Fnode> nodes;
    std::vector<double> powers;

    std::vector<std::vector<bool>> repetition_set;

    double * features;
    short n_features = 20;
    short * features_map;

    double longest_branch = -1000;

    int level = 0;


    Ftree(double * dd, int n_t, int mm,  std::vector<Edge> & edges_, double* feat=nullptr, short* feat_map=nullptr, bool nni_rep=false){
        d = dd;
        n_taxa = n_t;
        m = mm;
        edges = edges_;
        nni_repetition = nni_rep;

        subtree_distance = std::vector<std::vector<double>> (2*m - 2, std::vector<double>  (2*m - 2, 0));
        subtrees = std::vector<Edge> (2*edges.size(), Edge ());
        repetition_set = std::vector<std::vector<bool>> (2*edges.size(), std::vector<bool> (2*edges.size(), false));
        subtrees_n_nodes = std::vector<double> (2*edges.size(), 0);
        subtrees_max_length = std::vector<double> (2*edges.size(), 0);
        subtrees_total_length = std::vector<double> (2*edges.size(), 0);

        if(feat == nullptr) {
             if(not nni_rep) features = new double[(2 * n_taxa - 6) * (2 * n_taxa - 7) * n_features];
             else features = new double[(2 * n_taxa - 6) * (2 * n_taxa - 7) * n_features + 2 * (n_taxa - 3) * 3];
        }
        else features = feat;

        if(feat_map == nullptr) {
            if(not nni_rep) features_map = new short[(2 * n_taxa - 6) * (2 * n_taxa - 7) * 4];
            else features_map = new short[(2 * n_taxa - 6) * (2 * n_taxa - 7) * 4 + 2 * (n_taxa - 3) * 3];
            }
        else features_map = feat_map;


        T = std::vector<std::vector<short>> (m, std::vector<short>  (m, 0));
        nodes = std::map<int, Fnode> ();

        int power_2 = 1;
        powers = std::vector<double> (edges.size(), 1);

        Edge reverse_edge;
        // for(const Edge t : edges) std::cout<<get<0>(t)<<" "<<get<1>(t)<<std::endl;
        for(int i=0; i< edges.size(); i++){
            if (i > 0) powers[i] = powers[i - 1] * 0.5;
            // power_2 *= 2;
            edge_idxs[edges[i]] = i;
            subtrees[i] = edges[i];
            get_head(reverse_edge) = get_tail(edges[i]); get_tail(reverse_edge) = get_head(edges[i]);
            subtrees[i + edges.size()] = reverse_edge;
            edge_idxs[reverse_edge] = i + edges.size();
            if(nodes.find(get_head(edges[i]))==nodes.end()) {
                nodes[get_head(edges[i])] = Fnode (get_head(edges[i]), get_head(edges[i]) < n_taxa);
            }
            if(nodes.find(get_tail(edges[i]))==nodes.end()) {
                nodes[get_tail(edges[i])] = Fnode (get_tail(edges[i]), get_tail(edges[i]) < n_taxa);
            }

            nodes[get_head(edges[i])].add_link(nodes[get_tail(edges[i])]);
            nodes[get_tail(edges[i])].add_link(nodes[get_head(edges[i])]);

            }


        double dist;
        Edge current_edge; 
        Edge edge_b;
        for(Edge edge : edges) {
            if(get_head(edge) < n_taxa) {get_head(current_edge) = get_tail(edge); get_tail(current_edge) = get_head(edge);}
            else { current_edge = edge;}
            get_head(edge_b) = get_tail(current_edge); get_tail(edge_b) = get_head(current_edge);
            dist = compute_dist(current_edge, edge_b);
            // std::cout<<edge_idxs[current_edge]<<"   "<<edge_idxs[edge_b]<<std::endl;
            subtree_distance[edge_idxs[current_edge]][edge_idxs[edge_b]] = dist;
            subtree_distance[edge_idxs[edge_b]][edge_idxs[current_edge]] = dist;
            T[get_head(edge)][get_tail(edge)] = 1;
            T[get_tail(edge)][get_head(edge)] = 1;
            traverse_and_compute_dist(current_edge, edge_b, 1);
            traverse_and_compute_dist(edge_b, current_edge, 1);


        }

        compute_branch_length();

        tree_length = compute_tree_length();
        for(Edge edge : edges)  compute_subtree_properties(edge);


        compute_features();

    }
    ~Ftree(){
            features = nullptr;
            features_map = nullptr;
    }
    edge_tuple  get_children (int node, int parent) {
        int a, b;
        if(nodes[node].links[0]->index != parent) {
            a = nodes[node].links[0] ->index;
            if(nodes[node].links[1] ->index != parent) b = nodes[node].links[1] ->index;
            else b = nodes[node].links[2] ->index;
            }
        else {a = nodes[node].links[1] ->index; b = nodes[node].links[2] ->index;}
        edge_tuple children (Edge (node, a), Edge (node, b));
        return  children;
    }

    void set_feature(Edge &a, Edge &b, Edge &c, Edge &d, short step, double branch_dist, double dist_bc, double new_length) {
        
        // std::cout<<level<<"  "<<get_head(a)<<" "<<get_tail(a)<<" "<<get_head(d)<<" "<<get_tail(d)<<std::endl;
        features[level * n_features] = tree_length;
        features[level* n_features + 1] = longest_branch;
        features[level* n_features + 2] = get_length(a);
        features[level* n_features + 3] = get_length(d);
        features[level* n_features + 4] = step - 1;
        features[level* n_features + 5] = branch_dist;
        // dist_bc, new_length, *a_info, *b_info, *c_info, *d_info
        features[level* n_features + 6] = dist_bc;
        features[level* n_features + 7] = new_length;
        features[level* n_features + 8] = subtrees_n_nodes[edge_idxs[a]];
        features[level* n_features + 9] = subtrees_max_length[edge_idxs[a]];
        features[level* n_features + 10] = subtrees_total_length[edge_idxs[a]];
        features[level* n_features + 11] = subtrees_n_nodes[edge_idxs[b]];
        features[level* n_features + 12] = subtrees_max_length[edge_idxs[b]];
        features[level* n_features + 13] = subtrees_total_length[edge_idxs[b]];
        features[level* n_features + 14] = subtrees_n_nodes[edge_idxs[c]];
        features[level* n_features + 15] = subtrees_max_length[edge_idxs[c]];
        features[level* n_features + 16] = subtrees_total_length[edge_idxs[c]];
        features[level* n_features + 17] = subtrees_n_nodes[edge_idxs[d]];
        features[level* n_features + 18] = subtrees_max_length[edge_idxs[d]];
        features[level* n_features + 19] = subtrees_total_length[edge_idxs[d]];

        features_map[level*4] = get_head(a);
        features_map[level*4 + 1] = get_tail(a);
        features_map[level*4 + 2] = get_head(d);
        features_map[level*4 + 3] = get_tail(d);
        level++;
    }


    void compute_subtree_properties(Edge &edge){
        if (subtrees_n_nodes[edge_idxs[edge]] > 0) {return;}
        else{
            if(get_tail(edge) < n_taxa) {
                subtrees_n_nodes[edge_idxs[edge]] = 1;
                subtrees_max_length[edge_idxs[edge]] = get_length(edge);
                subtrees_total_length[edge_idxs[edge]] = get_length(edge);
            }
            else{
                edge_tuple children = get_children(get_tail(edge), get_head(edge));
                double n_subtrees = 0;
                double max_length = -1000;
                double total_length = 0;
                if (subtrees_n_nodes[edge_idxs[get<0> (children)]] == 0) compute_subtree_properties(get<0> (children));
                if (subtrees_n_nodes[edge_idxs[get<1> (children)]] == 0) compute_subtree_properties(get<1> (children));
                subtrees_n_nodes[edge_idxs[edge]] = subtrees_n_nodes[edge_idxs[get<0> (children)]] + subtrees_n_nodes[edge_idxs[get<1> (children)]];
                subtrees_total_length[edge_idxs[edge]] = get_length(edge) + subtrees_total_length[edge_idxs[get<0> (children)]] + subtrees_total_length[edge_idxs[get<1> (children)]];
                if(subtrees_max_length[edge_idxs[get<0> (children)]] > subtrees_max_length[edge_idxs[get<1> (children)]]){
                    subtrees_max_length[edge_idxs[edge]] = subtrees_max_length[edge_idxs[get<0> (children)]];
                }
                else{
                    subtrees_max_length[edge_idxs[edge]] = subtrees_max_length[edge_idxs[get<1> (children)]];
                }
                if (subtrees_max_length[edge_idxs[edge]] < get_length(edge)) subtrees_max_length[edge_idxs[edge]] = get_length(edge);

            }
        }
    }

    double compute_tree_length() {
        double length = 0;
        for ( const auto edge : edges ) {
            length += branch_length[edge];
            if(branch_length[edge] > longest_branch) longest_branch = branch_length[edge];
        }
        return length;
    }

    double dist(Edge &a, Edge &b) {return subtree_distance[edge_idxs[a]][edge_idxs[b]]; }


    double compute_dist(Edge &a, Edge &b){
        if(! dist(a, b) > 0){
            if(get_tail(a) < n_taxa){
                if(get_tail(b) < n_taxa) {
                    return d[get_tail(a)*n_taxa + get_tail(b)];
                    }
                else{
                    edge_tuple tail_links = get_children(get_tail(b), get_head(b));
                    return (compute_dist(a, get<0> (tail_links)) + compute_dist(a, get<1> (tail_links))) * 0.5;
                    }
                }
            else{
                if(get_tail(b) < n_taxa) {
                    edge_tuple tail_links = get_children(get_tail(a), get_head(a));
                    return (compute_dist(b, get<0> (tail_links)) + compute_dist(b, get<1> (tail_links))) * 0.5;
                    }
                else{
                    edge_tuple tail_links = get_children(get_tail(b), get_head(b));
                    return (compute_dist(a, get<0> (tail_links)) + compute_dist(a, get<1> (tail_links))) * 0.5;
                    }
                }
            }
        else {
            return dist(a, b);
            }
        }

    void traverse_and_compute_dist(Edge &a, Edge &b, short top_dist) {
        if(not nodes[get_tail(b)].leaf) {
            edge_tuple tail_links = get_children(get_tail(b), get_head(b));
            top_dist += 1;

            subtree_distance[edge_idxs[a]][edge_idxs[get<0> (tail_links)]] = compute_dist(a, get<0> (tail_links));
            subtree_distance[edge_idxs[get<0> (tail_links)]][edge_idxs[a]] = subtree_distance[edge_idxs[a]][edge_idxs[get<0> (tail_links)]];
            T[get_tail(a)][get_tail(get<0> (tail_links))] = top_dist;
            traverse_and_compute_dist(a, get<0> (tail_links), top_dist);

            subtree_distance[edge_idxs[a]][edge_idxs[get<1> (tail_links)]] = compute_dist(a, get<1> (tail_links));
            subtree_distance[edge_idxs[get<1> (tail_links)]][edge_idxs[a]] = subtree_distance[edge_idxs[a]][edge_idxs[get<1> (tail_links)]];
            T[get_tail(a)][get_tail(get<1> (tail_links))] = top_dist;
            traverse_and_compute_dist(a, get<1> (tail_links), top_dist);
        }

        // for(){
        //     subtree_distance[edge_idxs[current_edge]][edge_idxs[edge_b]]
    }

    void compute_branch_length(){
        edge_tuple tail_links;
        edge_tuple tail_links_1;
        Edge reverse_edge;

        for(Edge edge : edges)  {
            if (get_head(edge) < n_taxa) {
                tail_links = get_children(get_tail(edge), get_head(edge));
                get_head(reverse_edge) = get_tail(edge); get_tail(reverse_edge) = get_head(edge);
                branch_length[edge] =   (subtree_distance[edge_idxs[reverse_edge]][edge_idxs[get<0> (tail_links)]] +
                                        subtree_distance[edge_idxs[reverse_edge]][edge_idxs[get<1> (tail_links)]] -
                                        subtree_distance[edge_idxs[get<0> (tail_links)]][edge_idxs[get<1> (tail_links)]]) * 0.5;
            }
            else{
                if(get_tail(edge) < n_taxa) {
                    tail_links = get_children(get_head(edge), get_tail(edge));
                    branch_length[edge] =   (subtree_distance[edge_idxs[edge]][edge_idxs[get<0> (tail_links)]] +
                                            subtree_distance[edge_idxs[edge]][edge_idxs[get<1> (tail_links)]] -
                                            subtree_distance[edge_idxs[get<0> (tail_links)]][edge_idxs[get<1> (tail_links)]]) * 0.5;
                }
                else{
                    tail_links = get_children(get_head(edge), get_tail(edge));
                    tail_links_1 = get_children(get_tail(edge), get_head(edge));

                    branch_length[edge] =   (subtree_distance[edge_idxs[get<0> (tail_links)]][edge_idxs[get<0> (tail_links_1)]] +
                                            subtree_distance[edge_idxs[get<0> (tail_links)]][edge_idxs[get<1> (tail_links_1)]] +
                                            subtree_distance[edge_idxs[get<1> (tail_links)]][edge_idxs[get<0> (tail_links_1)]]+
                                            subtree_distance[edge_idxs[get<1> (tail_links)]][edge_idxs[get<1> (tail_links_1)]]) * 0.25 -
                                            (subtree_distance[edge_idxs[get<0> (tail_links)]][edge_idxs[get<1> (tail_links)]] +
                                            subtree_distance[edge_idxs[get<0> (tail_links_1)]][edge_idxs[get<1> (tail_links_1)]]) * 0.5;

                }
            }
        }

    }

    double get_length(Edge edge) {
        if (get_head(edge) < get_tail(edge)) {
            return branch_length[edge];
            }
        else {
            Edge reverse_edge = Edge (get_tail(edge), get_head(edge));
            return branch_length[reverse_edge];
            }

    }


    void traverse_and_compute_spr(Edge a, Edge c, Edge d, Edge b_0, Edge next_c, 
                                    double branch_dist, short step, double old_tree_length, 
                                    double dist_ab, double dist_bc) {
        
        step += 1;
        double new_length = old_tree_length + ((dist_bc + dist(a, d)) - (dist_ab + dist(c, d))) * 0.25;
        branch_dist += get_length(d);

        double new_dist_ab = (dist_ab + dist(a, c)) * 0.5;

        if(step==1 and not nni_repetition){
            if(not repetition_set[edge_idxs[a]][edge_idxs[d]]) {
                set_feature(a, b_0, c, d, step, branch_dist, dist_bc, new_length);
                repetition_set[edge_idxs[a]][edge_idxs[d]] = true;
                repetition_set[edge_idxs[d]][edge_idxs[a]] = true;
                repetition_set[edge_idxs[b_0]][edge_idxs[c]] = true;
                repetition_set[edge_idxs[c]][edge_idxs[b_0]] = true;
            }            
        }
        else set_feature(a, b_0, c, d, step, branch_dist, dist_bc, new_length);

        if(not (get_tail(d) < n_taxa)) {
            edge_tuple children = get_children(get_tail(d), get_head(d));
            Edge d_new = get<0>(children);
            Edge c_new = get<1> (children);
            double new_dist_bc = dist(next_c, c_new)  + powers[step + 1] * ( dist(b_0, c_new) - dist(a, c_new));

            Edge new_next_c = Edge (get_tail(d_new), get_head(d_new));
            
            traverse_and_compute_spr(a, c_new, d_new, b_0, new_next_c, branch_dist, step, new_length,
                                new_dist_ab, new_dist_bc);

            d_new = get<1>(children);
            c_new = get<0> (children);
            new_dist_bc = dist(next_c, c_new)  +  powers[step + 1] * (dist(b_0, c_new) - dist(a, c_new));

            new_next_c = Edge (get_tail(d_new), get_head(d_new));
            
            traverse_and_compute_spr(a, c_new, d_new, b_0, new_next_c, branch_dist, step, new_length,
                                new_dist_ab, new_dist_bc);

        }


    }

    void compute_features() {
        short step = 0;
        edge_tuple tail_links;
        edge_tuple children;
        Edge b0;
        Edge c;
        Edge d;
        Edge next_c;
        short i, j;
        short level = 0;
        double old_dist_ab, old_dist_bc, branch_dist;

        for(Edge a : subtrees) {
            if(not (get_head(a) < n_taxa)){

                tail_links = get_children(get_head(a), get_tail(a));

                if( not (get_tail(get<0> (tail_links)) < n_taxa)) {
                    children = get_children(get_tail(get<0> (tail_links)), get_head(get<0> (tail_links)));

                    b0 = get<1> (tail_links);
                    branch_dist = get_length(get<0> (tail_links));
                    // children 1  c = get<1> (tail_links)
                    d = get<0> (children);
                    old_dist_ab = dist(a, b0);
                    old_dist_bc = dist(b0, get<1> (children));
                    get_head(next_c) = get_tail(d); get_tail(next_c) = get_head(d);
                    traverse_and_compute_spr(a, get<1> (children), d, b0, next_c, branch_dist, step, tree_length, old_dist_ab, old_dist_bc);


                    // children 2 c = get<0> (children)
                    d = get<1> (children);
                    old_dist_ab = dist(a, b0);
                    old_dist_bc = dist(b0, get<0> (children));
                    get_head(next_c) = get_tail(d); get_tail(next_c) = get_head(d);
                    
                    traverse_and_compute_spr(a, get<0> (children), d, b0, next_c, branch_dist, step, tree_length, old_dist_ab, old_dist_bc);

                }

                if( not (get_tail(get<1> (tail_links)) < n_taxa)) {
                    children = get_children(get_tail(get<1> (tail_links)), get_head(get<1> (tail_links)));

                    b0 = get<0> (tail_links);
                    branch_dist = get_length(get<1> (tail_links));
                    // children 1  c = get<1> (tail_links)
                    d = get<0> (children);
                    old_dist_ab = dist(a, b0);
                    old_dist_bc = dist(b0, get<1> (children));
                    get_head(next_c) = get_tail(d); get_tail(next_c) = get_head(d);
                    
                    traverse_and_compute_spr(a, get<1> (children), d, b0, next_c, branch_dist, step, tree_length, old_dist_ab, old_dist_bc);


                    // children 2 c = get<0> (children)
                    d = get<1> (children);
                    old_dist_ab = dist(a, b0);
                    old_dist_bc = dist(b0, get<0> (children));
                    get_head(next_c) = get_tail(d); get_tail(next_c) = get_head(d);
                    
                    traverse_and_compute_spr(a, get<0> (children), d, b0, next_c, branch_dist, step, tree_length, old_dist_ab, old_dist_bc);

                }
 
            }
        }

    }

};

